#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_video_out_overlay_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/windowInternal.h"
#include "libs/hmd2.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>

// A private PNG writer used only for opt-in Vulkan framebuffer diagnostics.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBIWDEF static inline
#include "stb_image_write.h"
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

namespace Libs::Graphics {
std::atomic_bool g_vr_eye_crop {true};


namespace {
constexpr uint64_t kPreviewSeriesStart = 220u;
constexpr uint64_t kPreviewSeriesFrames = 75u;

const std::vector<uint32_t>& PreviewImageSeries() {
    static const std::vector<uint32_t> ids = [] {
        std::vector<uint32_t> values;
        const char* next = std::getenv("KYTY_VR_PREVIEW_IDS");
        if (next == nullptr) {
            return values;
        }
        while (*next != '\0' && values.size() < 24u) {
            char* end = nullptr;
            const auto id = std::strtoul(next, &end, 10);
            if (end == next || id > 100000u) {
                break;
            }
            values.push_back(static_cast<uint32_t>(id));
            next = *end == ',' ? end + 1 : end;
            if (*end == '\0') {
                break;
            }
        }
        return values;
    }();
    return ids;
}

size_t PreviewSeriesIndex(uint64_t frame) {
    const auto& ids = PreviewImageSeries();
    if (ids.empty() || frame < kPreviewSeriesStart) {
        return 0;
    }
    return std::min(static_cast<size_t>((frame - kPreviewSeriesStart) /
                                         kPreviewSeriesFrames),
                    ids.size() - 1u);
}

int WriteDiagnosticPng(const std::string& path, uint32_t width, uint32_t height,
                       vk::Format format, const uint8_t* raw, bool opaque = false) {
    const auto bytes = static_cast<uint64_t>(width) * height * 4u;
    std::vector<uint8_t> rgba(static_cast<size_t>(bytes));
    if (format == vk::Format::eA2B10G10R10UnormPack32 ||
        format == vk::Format::eA2R10G10B10UnormPack32) {
        for (size_t i = 0; i < rgba.size(); i += 4) {
            uint32_t word = 0;
            std::memcpy(&word, raw + i, 4);
            const uint32_t lo = (word & 0x3ffu) * 255u / 1023u;
            const uint32_t mid = ((word >> 10u) & 0x3ffu) * 255u / 1023u;
            const uint32_t hi = ((word >> 20u) & 0x3ffu) * 255u / 1023u;
            rgba[i] = static_cast<uint8_t>(
                format == vk::Format::eA2B10G10R10UnormPack32 ? lo : hi);
            rgba[i + 1] = static_cast<uint8_t>(mid);
            rgba[i + 2] = static_cast<uint8_t>(
                format == vk::Format::eA2B10G10R10UnormPack32 ? hi : lo);
            rgba[i + 3] = 255;
        }
    } else if (format == vk::Format::eB8G8R8A8Unorm ||
               format == vk::Format::eB8G8R8A8Srgb) {
        for (size_t i = 0; i < rgba.size(); i += 4) {
            rgba[i] = raw[i + 2];
            rgba[i + 1] = raw[i + 1];
            rgba[i + 2] = raw[i];
            rgba[i + 3] = raw[i + 3];
        }
    } else {
        std::memcpy(rgba.data(), raw, rgba.size());
    }
    if (opaque) {
        for (size_t i = 3; i < rgba.size(); i += 4) {
            rgba[i] = 255;
        }
    }
    return stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height),
                          4, rgba.data(), static_cast<int>(width * 4u));
}
} // namespace
struct Presenter::Frame {
	VulkanImage   image;
	vk::ImageView view         = nullptr;
	uint64_t      present_tick = 0;
	bool          busy         = false;

	void Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
	void Transit(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access);
	void CopyFrom(CommandBuffer& command, Image& source);
	void BlitStereo(CommandBuffer& command, Image& left, Image& right);
	void Clear(CommandBuffer& command, const vk::ClearColorValue& color);
};

class FramePool final {
public:
	FramePool(WindowContext& window, CommandScheduler& scheduler)
	    : m_window(window), m_scheduler(scheduler) {}
	~FramePool() {
		m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
		for (auto& frame: m_frames) {
			m_window.graphic_ctx.device.destroyImageView(frame->view, nullptr);
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
		}
	}
	KYTY_CLASS_NO_COPY(FramePool);

	void Initialize(uint32_t count, vk::Format format) {
		if (count == 0 || format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires at least one frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!m_frames.empty()) {
			EXIT("prepared-frame pool was initialized twice\n");
		}
		m_format = format;
		m_frames.reserve(count);
		m_free.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
	}

	void SetFormat(vk::Format format) {
		if (format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires a presentation format\n");
		}
		Common::LockGuard lock(m_mutex);
		m_format = format;
	}

	vk::Format GetFormat() {
		Common::LockGuard lock(m_mutex);
		if (m_format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool has no presentation format\n");
		}
		return m_format;
	}

	Presenter::Frame* Acquire(vk::Extent2D extent, vk::Format format) {
		m_mutex.Lock();
		if (m_frames.empty()) {
			EXIT("prepared-frame pool was used before swapchain initialization\n");
		}
		// A synchronized flip may need more frames than the swapchain has images.
		// Queue capacity bounds growth; waiting here can prevent its master from recording.
		if (m_free.empty()) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
		const auto compatible = std::ranges::find_if(m_free, [&](const auto* frame) {
			return frame->image.extent.width == extent.width &&
			       frame->image.extent.height == extent.height && frame->image.format == format;
		});
		auto*      frame      = compatible == m_free.end() ? m_free.back() : *compatible;
		if (compatible != m_free.end()) {
			*compatible = m_free.back();
		}
		m_free.pop_back();
		if (frame->busy) {
			EXIT("prepared-frame pool returned an invalid frame\n");
		}
		frame->busy = true;
		m_mutex.Unlock();

		m_scheduler.Wait(frame->present_tick);
		return frame;
	}

	void ValidateForPresent(Presenter::Frame* frame) {
		Common::LockGuard lock(m_mutex);
		if (frame == nullptr || !frame->busy) {
			EXIT("prepared frame has invalid presentation ownership\n");
		}
	}

	void Release(Presenter::Frame* frame) {
		if (frame == nullptr) {
			EXIT("cannot release a null prepared frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!frame->busy) {
			EXIT("prepared frame was released twice\n");
		}
		frame->busy = false;
		m_free.push_back(frame);
	}

private:
	WindowContext&                                 m_window;
	CommandScheduler&                              m_scheduler;
	Common::Mutex                                  m_mutex;
	std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
	std::vector<Presenter::Frame*>                 m_free;
	vk::Format                                     m_format = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return;
	}

	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferSrc |
	    vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
		     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
	}

	if (dst.image != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
		graphics.DeleteImage(dst);
	}

	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
	               vk::ImageUsageFlagBits::eSampled;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
		     extent.width, extent.height, static_cast<int>(format));
	}
}

void Presenter::Frame::Transit(vk::CommandBuffer command, vk::ImageLayout layout,
                               vk::AccessFlags2 access) {
	const auto     stage  = access == vk::AccessFlagBits2::eTransferRead ||
	                                access == vk::AccessFlagBits2::eTransferWrite
	                            ? vk::PipelineStageFlagBits2::eTransfer
	                            : vk::PipelineStageFlagBits2::eAllCommands;
	constexpr auto writes = vk::AccessFlagBits2::eTransferWrite |
	                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
	if (image.state.layout == layout && image.state.access_mask == access &&
	    !static_cast<bool>(image.state.access_mask & writes)) {
		return;
	}
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                    = image.state.pl_stage;
	barrier.srcAccessMask                   = image.state.access_mask;
	barrier.dstStageMask                    = stage;
	barrier.dstAccessMask                   = access;
	barrier.oldLayout                       = image.state.layout;
	barrier.newLayout                       = layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image.image;
	barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
	image.subresource_states.clear();
}

void Presenter::Frame::CopyFrom(CommandBuffer& command_buffer, Image& source) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, source.backing.layers};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, image.layers};
	copy.extent         = {std::min(source.backing.extent.width, image.extent.width),
	                       std::min(source.backing.extent.height, image.extent.height), 1};
	EXIT_IF(copy.srcSubresource.layerCount != copy.dstSubresource.layerCount);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
}

void Presenter::Frame::BlitStereo(CommandBuffer& command_buffer, Image& left, Image& right) {
    // Opt-in diagnostic: assemble 2 native eye render targets into a
    // side-by-side desktop image without lens distortion or timewarp.
    command_buffer.EndRendering();
    const auto command = command_buffer.Handle();
    Transit(command, vk::ImageLayout::eTransferDstOptimal,
            vk::AccessFlagBits2::eTransferWrite);
    const std::array<Image*, 2> sources {&left, &right};
    for (size_t eye = 0; eye < sources.size(); ++eye) {
        auto& src = *sources[eye];
        if (src.backing.layers != 1u || src.backing.extent.depth != 1u) {
            EXIT("stereo preview source has unsupported layers or depth\n");
        }
        src.Transit(vk::ImageLayout::eTransferSrcOptimal,
                    vk::AccessFlagBits2::eTransferRead, {}, command);
        vk::ImageBlit region {};
        region.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region.srcOffsets[0] = {0, 0, 0};
        region.srcOffsets[1] = {static_cast<int32_t>(src.backing.extent.width),
                                static_cast<int32_t>(src.backing.extent.height), 1};
        const auto width = static_cast<int32_t>(image.extent.width);
        const auto height = static_cast<int32_t>(image.extent.height);
        region.dstOffsets[0] = {static_cast<int32_t>(eye) * (width / 2), 0, 0};
        region.dstOffsets[1] = {static_cast<int32_t>(eye + 1u) * (width / 2),
                                height, 1};
        command.blitImage(src.backing.image, vk::ImageLayout::eTransferSrcOptimal,
                          image.image, vk::ImageLayout::eTransferDstOptimal,
                          1, &region, vk::Filter::eNearest);
    }
}
void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
}

class Swapchain final {
public:
	enum class Status : uint8_t { Success, Recreate, SurfaceLost };

	explicit Swapchain(WindowContext& window): m_window(window) {}
	~Swapchain();
	KYTY_CLASS_NO_COPY(Swapchain);

	void                 Create();
	void                 Recreate(bool surface_lost = false);
	[[nodiscard]] bool   NeedsResize() const;
	[[nodiscard]] Status AcquireNextImage();
	[[nodiscard]] bool   PrepareSystemOverlay();
	void     RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
	                               const Presenter::Layer& overlay, bool draw_system_overlay);
	uint64_t Submit(CommandScheduler& scheduler);
	void     FinishPresentCapture(CommandScheduler& scheduler, uint64_t tick);
	[[nodiscard]] Status Present();

	[[nodiscard]] uint32_t ImageCount() const noexcept {
		return static_cast<uint32_t>(m_images.size());
	}
	[[nodiscard]] vk::Format Format() const noexcept { return m_format; }

private:
	void Destroy();
	void DrawOverlay(vk::CommandBuffer command, const Presenter::Layer& layer);
	void RecordPresentCapture(vk::CommandBuffer command);
	void ReleasePresentCapture();

	WindowContext&   m_window;
	vk::SwapchainKHR m_handle = nullptr;
	vk::Format       m_format = vk::Format::eUndefined;
	vk::Extent2D     m_extent {};
	// Drawable pixel size observed when this swapchain was created.
	vk::Extent2D                   m_window_extent {};
	std::vector<vk::Image>         m_images;
	std::vector<vk::ImageView>     m_image_views;
	std::vector<vk::Semaphore>     m_image_acquired;
	std::vector<vk::Semaphore>     m_render_complete;
	std::unique_ptr<SystemOverlay> m_system_overlay;
	vk::DescriptorSetLayout        m_overlay_descriptors = nullptr;
	vk::PipelineLayout             m_overlay_layout      = nullptr;
	vk::Pipeline                   m_overlay_pipeline    = nullptr;
	vk::Sampler                    m_overlay_sampler     = nullptr;
	uint32_t                       m_image_index         = static_cast<uint32_t>(-1);
	uint32_t                       m_frame_index         = 0;
	bool                           m_present_capture_supported = false;
	bool                           m_present_capture_warned    = false;
	bool                           m_present_capture_done      = false;
	size_t                         m_present_capture_index     = 0;
	VkBuffer                       m_present_capture_buffer    = VK_NULL_HANDLE;
	VmaAllocation                  m_present_capture_allocation = nullptr;
	std::string                    m_present_capture_path;
	vk::Extent2D                   m_present_capture_extent {};
	vk::Format                     m_present_capture_format = vk::Format::eUndefined;
	uint64_t                       m_present_capture_frame  = 0;
};

struct Presenter::Impl {
	explicit Impl(WindowContext& owner)
	    : renderer(*owner.render_context), window(owner), swapchain(owner),
	      present_scheduler(renderer, owner.graphic_ctx), frames(owner, present_scheduler) {
		EXIT_IF(owner.render_context == nullptr);
		swapchain.Create();
		frames.Initialize(swapchain.ImageCount(), swapchain.Format());
	}

	void RecoverSwapchain(Swapchain::Status status) {
		LOGF("Recovering Vulkan swapchain%s\n",
		     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
		swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
		frames.SetFormat(swapchain.Format());
	}

	Image& ResolveSurface(const ImageInfo& info, const ImageId* direct_source = nullptr) {
		TextureCache::ImageDesc desc {};
		desc.info                  = info;
		desc.view_info.format      = info.pixel_format;
		desc.view_info.type        = vk::ImageViewType::e2D;
		desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		desc.view_info.base_level  = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer  = 0;
		desc.view_info.layer_count = 1;
		desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
		desc.type                  = TextureCache::BindingType::VideoOut;

		auto&      cache      = renderer.GetTextureCache();
		if (direct_source != nullptr) {
			auto& image = cache.GetImage(*direct_source);
			image.usage.video_out = true;
			cache.UpdateImage(*direct_source);
			return image;
		}
        if (std::getenv("KYTY_VR_DUMP_IMAGES") != nullptr) {
            static bool reported = false;
            if (!reported && window.loop.presented_frames.load(std::memory_order_relaxed) >= 450) {
                reported = true;
                cache.DumpReprojectionCandidates();
            }
        }
		const auto image_id   = cache.FindImage(desc);
		auto&      image      = cache.GetImage(image_id);
		image.usage.video_out = true;
		cache.UpdateImage(image_id);
		return image;
	}
	void Present();
	void CaptureFrameIfRequested(Presenter::Frame* frame);

	RenderContext&        renderer;
	WindowContext&        window;
	Swapchain             swapchain;
	CommandScheduler      present_scheduler;
	FramePool             frames;
	Common::Mutex         present_mutex;
	std::array<Layer, 2>  layers {};
	std::atomic<uint64_t> presented_overlay_revision {0};
};

void Swapchain::Create() {
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	// Capture the size before querying surface capabilities. If the window changes
	// during creation, the next presentation will detect the newer size.
	{
		Common::LockGuard lock(m_window.mutex);
		m_window_extent = {graphics.screen_width, graphics.screen_height};
	}
	EXIT_IF(m_window_extent.width == 0);
	EXIT_IF(m_window_extent.height == 0);
	m_window.RefreshSurfaceCapabilities();
	const auto& surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	m_extent = surface.capabilities.currentExtent;
	if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
		m_extent.width =
		    std::clamp(m_window_extent.width, surface.capabilities.minImageExtent.width,
		               surface.capabilities.maxImageExtent.width);
		m_extent.height =
		    std::clamp(m_window_extent.height, surface.capabilities.minImageExtent.height,
		               surface.capabilities.maxImageExtent.height);
	}
	uint32_t image_count = surface.capabilities.minImageCount + 1;
	if (surface.capabilities.maxImageCount != 0) {
		image_count = std::min(image_count, surface.capabilities.maxImageCount);
	}
	const auto transform =
	    surface.capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
	        ? vk::SurfaceTransformFlagBitsKHR::eIdentity
	        : surface.capabilities.currentTransform;
	const auto composite =
	    surface.capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
	        ? vk::CompositeAlphaFlagBitsKHR::eOpaque
	        : vk::CompositeAlphaFlagBitsKHR::eInherit;

	vk::SurfaceFormatKHR format {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
	if (surface.formats.size() != 1 || surface.formats.front().format != vk::Format::eUndefined) {
		const auto it = std::find_if(surface.formats.begin(), surface.formats.end(),
		                             [](const vk::SurfaceFormatKHR& candidate) {
			                             return candidate.format == vk::Format::eB8G8R8A8Unorm ||
			                                    candidate.format == vk::Format::eR8G8B8A8Unorm;
		                             });
		if (it == surface.formats.end()) {
			EXIT("no supported UNORM swapchain format\n");
		}
		format = *it;
	}
	m_format                      = format.format;
	const auto swapchain_features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
	if (!static_cast<bool>(swapchain_features & vk::FormatFeatureFlagBits::eBlitDst)) {
		EXIT("swapchain format cannot be a blit destination: format=%d\n",
		     static_cast<int>(m_format));
	}

	vk::SwapchainCreateInfoKHR create_info {};
	create_info.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
	create_info.surface          = m_window.surface;
	create_info.minImageCount    = image_count;
	create_info.imageFormat      = format.format;
	create_info.imageColorSpace  = format.colorSpace;
	create_info.imageExtent      = m_extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage =
	    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	m_present_capture_supported = false;
	if (const char* capture = std::getenv("KYTY_TEST_PRESENT_CAPTURE");
	    capture != nullptr && capture[0] != '\0') {
		m_present_capture_supported = static_cast<bool>(
		    surface.capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferSrc);
		if (m_present_capture_supported) {
			create_info.imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
		} else if (!m_present_capture_warned) {
			m_present_capture_warned = true;
			LOGF("Present frame capture unavailable: surface lacks transfer-source usage\n");
		}
	}
	create_info.imageSharingMode = vk::SharingMode::eExclusive;
	create_info.preTransform     = transform;
	create_info.compositeAlpha   = composite;
	switch (Config::GetPresentMode()) {
		case Config::PresentMode::Mailbox:
			create_info.presentMode = vk::PresentModeKHR::eMailbox;
			break;
		case Config::PresentMode::Immediate:
			create_info.presentMode = vk::PresentModeKHR::eImmediate;
			break;
		case Config::PresentMode::Fifo:
		default: create_info.presentMode = vk::PresentModeKHR::eFifo; break;
	}
	if (std::find(surface.present_modes.begin(), surface.present_modes.end(),
	              create_info.presentMode) == surface.present_modes.end()) {
		LOGF("warning: requested present mode is unavailable; falling back to Fifo\n");
		create_info.presentMode = vk::PresentModeKHR::eFifo;
	}
	create_info.clipped = VK_TRUE;
	RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
	                     "vkCreateSwapchainKHR");
	EXIT_IF(m_handle == nullptr);

	m_images = EnumerateVulkan<vk::Image>(
	    "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
		    return graphics.device.getSwapchainImagesKHR(m_handle, count, images);
	    });
	EXIT_NOT_IMPLEMENTED(m_images.empty());

	m_image_views.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		vk::ImageViewCreateInfo view {};
		view.sType                           = vk::StructureType::eImageViewCreateInfo;
		view.image                           = m_images[i];
		view.viewType                        = vk::ImageViewType::e2D;
		view.format                          = m_format;
		view.components                      = {};
		view.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.baseMipLevel   = 0;
		view.subresourceRange.layerCount     = 1;
		view.subresourceRange.levelCount     = 1;
		RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_image_views[i]),
		                     "vkCreateImageView");
		EXIT_IF(m_image_views[i] == nullptr);
	}

	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.sType = vk::StructureType::eSemaphoreCreateInfo;
	m_image_acquired.resize(m_images.size());
	m_render_complete.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_image_acquired[i]),
		    "create swapchain image-acquired semaphore");
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_render_complete[i]),
		    "create swapchain render-complete semaphore");
	}
	m_image_index = static_cast<uint32_t>(-1);
	m_frame_index = 0;
}

Swapchain::~Swapchain() {
	Destroy();
}

void Swapchain::Destroy() {
	if (m_handle == nullptr && m_image_acquired.empty() && m_render_complete.empty() &&
	    m_image_views.empty() && m_present_capture_buffer == VK_NULL_HANDLE) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;

	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.waitIdle(), "wait for swapchain queue");
	}
	ReleasePresentCapture();
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
	}
	graphics.device.destroyPipeline(m_overlay_pipeline, nullptr);
	graphics.device.destroyPipelineLayout(m_overlay_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_overlay_descriptors, nullptr);
	graphics.device.destroySampler(m_overlay_sampler, nullptr);
	m_overlay_pipeline    = nullptr;
	m_overlay_layout      = nullptr;
	m_overlay_descriptors = nullptr;
	m_overlay_sampler     = nullptr;

	for (const auto semaphore: m_image_acquired) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto semaphore: m_render_complete) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto view: m_image_views) {
		if (view != nullptr) {
			graphics.device.destroyImageView(view, nullptr);
		}
	}
	if (m_handle != nullptr) {
		graphics.device.destroySwapchainKHR(m_handle, nullptr);
	}

	m_handle        = nullptr;
	m_format        = vk::Format::eUndefined;
	m_extent        = {};
	m_window_extent = {};
	m_image_index   = static_cast<uint32_t>(-1);
	m_frame_index   = 0;
	m_images.clear();
	m_image_views.clear();
	m_image_acquired.clear();
	m_render_complete.clear();
}

void Swapchain::Recreate(bool surface_lost) {
	Destroy();
	if (surface_lost) {
#if defined(__APPLE__)
		// Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
		// window's view/layer and must run on the main thread on macOS.
		EXIT_IF(!SDL_RunOnMainThread(
		    [](void* window) { static_cast<WindowContext*>(window)->RecreateSurface(); }, &m_window,
		    true));
#else
		m_window.RecreateSurface();
#endif
	}
	Create();
}

bool Swapchain::NeedsResize() const {
	Common::LockGuard lock(m_window.mutex);
	return m_window_extent.width != m_window.graphic_ctx.screen_width ||
	       m_window_extent.height != m_window.graphic_ctx.screen_height;
}

Swapchain::Status Swapchain::AcquireNextImage() {
	EXIT_IF(m_handle == nullptr || m_frame_index >= m_image_acquired.size());
	m_image_index     = static_cast<uint32_t>(-1);
	const auto result = m_window.graphic_ctx.device.acquireNextImageKHR(
	    m_handle, std::numeric_limits<uint64_t>::max(), m_image_acquired[m_frame_index], nullptr,
	    &m_image_index);
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorUnknown:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorUnknown\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkAcquireNextImageKHR failed: %s\n", vk::to_string(result).c_str());
	}
	EXIT_IF(m_image_index >= m_images.size());
	return Status::Success;
}

bool Swapchain::PrepareSystemOverlay() {
	if (m_system_overlay == nullptr) {
		m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
	}
	return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount());
}

void Swapchain::DrawOverlay(vk::CommandBuffer command, const Presenter::Layer& layer) {
	auto device = m_window.graphic_ctx.device;
	if (m_overlay_pipeline == nullptr) {
		vk::SamplerCreateInfo sampler {};
		sampler.magFilter    = vk::Filter::eLinear;
		sampler.minFilter    = vk::Filter::eLinear;
		sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
		sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
		RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_overlay_sampler),
		                     "create video-out overlay sampler");
		const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eCombinedImageSampler,
		                                              1, vk::ShaderStageFlagBits::eFragment};
		vk::DescriptorSetLayoutCreateInfo    descriptors {};
		descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptors.bindingCount = 1;
		descriptors.pBindings    = &binding;
		RequireVulkanSuccess(
		    device.createDescriptorSetLayout(&descriptors, nullptr, &m_overlay_descriptors),
		    "create video-out overlay descriptor layout");
		const vk::PushConstantRange alpha {vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &m_overlay_descriptors;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &alpha;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_overlay_layout),
		                     "create video-out overlay pipeline layout");
		const auto vertex   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
		const auto fragment = CompileSPV(GPU_VIDEO_OUT_OVERLAY_SPV, device);
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
		stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
		stages[0].module = vertex;
		stages[0].pName  = "main";
		stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
		stages[1].module = fragment;
		stages[1].pName  = "main";
		vk::PipelineVertexInputStateCreateInfo   vertex_input {};
		vk::PipelineInputAssemblyStateCreateInfo assembly {};
		assembly.topology = vk::PrimitiveTopology::eTriangleList;
		vk::PipelineViewportStateCreateInfo viewport {};
		viewport.viewportCount = 1;
		viewport.scissorCount  = 1;
		vk::PipelineRasterizationStateCreateInfo rasterization {};
		rasterization.lineWidth = 1.0f;
		vk::PipelineMultisampleStateCreateInfo multisample {};
		multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
		vk::PipelineColorBlendAttachmentState attachment {};
		attachment.blendEnable         = VK_TRUE;
		attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
		attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
		                            vk::ColorComponentFlagBits::eG |
		                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		vk::PipelineColorBlendStateCreateInfo blend {};
		blend.attachmentCount = 1;
		blend.pAttachments    = &attachment;
		const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamic {};
		dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic.pDynamicStates    = dynamic_states.data();
		vk::PipelineRenderingCreateInfo rendering {};
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &m_format;
		vk::GraphicsPipelineCreateInfo create {};
		create.pNext               = &rendering;
		create.stageCount          = static_cast<uint32_t>(stages.size());
		create.pStages             = stages.data();
		create.pVertexInputState   = &vertex_input;
		create.pInputAssemblyState = &assembly;
		create.pViewportState      = &viewport;
		create.pRasterizationState = &rasterization;
		create.pMultisampleState   = &multisample;
		create.pColorBlendState    = &blend;
		create.pDynamicState       = &dynamic;
		create.layout              = m_overlay_layout;
		RequireVulkanSuccess(
		    device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &m_overlay_pipeline),
		    "create video-out overlay pipeline");
		device.destroyShaderModule(fragment, nullptr);
		device.destroyShaderModule(vertex, nullptr);
	}
	auto& frame = *layer.frame;
	if (frame.view == nullptr) {
		vk::ImageViewCreateInfo view {};
		view.image            = frame.image.image;
		view.viewType         = vk::ImageViewType::e2D;
		view.format           = frame.image.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		RequireVulkanSuccess(device.createImageView(&view, nullptr, &frame.view),
		                     "create video-out overlay image view");
	}
	const vk::DescriptorImageInfo image {m_overlay_sampler, frame.view,
	                                     vk::ImageLayout::eShaderReadOnlyOptimal};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	write.pImageInfo      = &image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_overlay_layout, 0, 1, &write);
	const uint32_t premultiplied = layer.premultiplied_alpha;
	command.pushConstants(m_overlay_layout, vk::ShaderStageFlagBits::eFragment, 0,
	                      sizeof(premultiplied), &premultiplied);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_overlay_pipeline);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = m_image_views[m_image_index];
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = m_extent;
	rendering.layerCount           = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	const vk::Viewport viewport {
	    0.0f, 0.0f, static_cast<float>(m_extent.width), static_cast<float>(m_extent.height),
	    0.0f, 1.0f};
	const vk::Rect2D scissor {{0, 0}, m_extent};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
                                      const Presenter::Layer& overlay, bool draw_system_overlay) {
	EXIT_IF(m_image_index >= m_images.size());
	auto       vk_command      = command.Handle();
	const bool draw_overlay    = overlay.frame != nullptr;
	const bool draw_attachment = draw_overlay || draw_system_overlay;
	if (source != nullptr) {
		source->Transit(vk_command, vk::ImageLayout::eTransferSrcOptimal,
		                vk::AccessFlagBits2::eTransferRead);
	}
	if (draw_overlay) {
		overlay.frame->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
		                       vk::AccessFlagBits2::eShaderRead);
	}

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType                           = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask                   = vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout                       = vk::ImageLayout::eUndefined;
	to_transfer.newLayout                       = vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = m_images[m_image_index];
	to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_transfer.subresourceRange.baseMipLevel   = 0;
	to_transfer.subresourceRange.levelCount     = 1;
	to_transfer.subresourceRange.baseArrayLayer = 0;
	to_transfer.subresourceRange.layerCount     = 1;
	// Match the acquire wait stage so the layout transition cannot precede acquisition.
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                           vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
	                           nullptr, 0, nullptr, 1, &to_transfer);

	if (source != nullptr) {
		vk::ImageBlit region {};
		region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.srcSubresource.mipLevel       = 0;
		region.srcSubresource.baseArrayLayer = 0;
		region.srcSubresource.layerCount     = 1;
		region.srcOffsets[1].x               = static_cast<int>(source->image.extent.width);
		if (Config::VrEnabled() && g_vr_eye_crop.load(std::memory_order_relaxed) &&
		    source->image.extent.width == Hmd2::PANEL_WIDTH &&
		    source->image.extent.height == Hmd2::PANEL_HEIGHT) {
			// A virtual-headset frame contains the stereo panel; center the
			// left eye in the desktop window rather than scaling both eyes.
			const int eye_width = static_cast<int>(source->image.extent.width / 2u);
			const int offset = static_cast<int>(source->image.extent.width / 16u);
			region.srcOffsets[0].x = offset;
			region.srcOffsets[1].x = offset + eye_width;
		}
		region.srcOffsets[1].y               = static_cast<int>(source->image.extent.height);
		region.srcOffsets[1].z               = 1;
		region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
		region.dstSubresource.mipLevel       = 0;
		region.dstSubresource.baseArrayLayer = 0;
		region.dstSubresource.layerCount     = 1;
		region.dstOffsets[1].x               = static_cast<int>(m_extent.width);
		region.dstOffsets[1].y               = static_cast<int>(m_extent.height);
		region.dstOffsets[1].z               = 1;
		vk_command.blitImage(source->image.image, vk::ImageLayout::eTransferSrcOptimal,
		                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1,
		                     &region, vk::Filter::eLinear);
	} else {
		const vk::ClearColorValue black {};
		vk_command.clearColorImage(m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal,
		                           &black, 1, &to_transfer.subresourceRange);
	}

	vk::ImageMemoryBarrier to_present {};
	to_present.sType               = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask       = draw_attachment ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                       vk::AccessFlagBits::eColorAttachmentWrite
	                                                 : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout           = vk::ImageLayout::eTransferDstOptimal;
	to_present.newLayout           = draw_attachment ? vk::ImageLayout::eColorAttachmentOptimal
	                                                 : vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_present.image               = m_images[m_image_index];
	to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_present.subresourceRange.baseMipLevel   = 0;
	to_present.subresourceRange.levelCount     = 1;
	to_present.subresourceRange.baseArrayLayer = 0;
	to_present.subresourceRange.layerCount     = 1;
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                           draw_attachment ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                                           : vk::PipelineStageFlagBits::eAllCommands,
	                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
	                           &to_present);
	if (draw_attachment) {
		if (draw_overlay) {
			DrawOverlay(vk_command, overlay);
			if (draw_system_overlay) {
				to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
				to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
				vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0,
				                           nullptr, 1, &to_present);
			}
		}
		if (draw_system_overlay) {
			m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
		}
		to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
		to_present.newLayout     = vk::ImageLayout::ePresentSrcKHR;
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                           vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
	RecordPresentCapture(vk_command);
}

void Swapchain::RecordPresentCapture(vk::CommandBuffer command) {
	if (!m_present_capture_supported) {
		return;
	}
	const char* filename = std::getenv("KYTY_TEST_PRESENT_CAPTURE");
	if (filename == nullptr || filename[0] == '\0') {
		return;
	}
	const auto index = m_window.loop.presented_frames.load(std::memory_order_relaxed);
	std::string capture_path(filename);
	if (const char* points = std::getenv("KYTY_TEST_CAPTURE_POINTS");
	    points != nullptr && points[0] != '\0') {
		static const std::vector<uint64_t> thresholds = [] {
			std::vector<uint64_t> values;
			const char* next = std::getenv("KYTY_TEST_CAPTURE_POINTS");
			while (next != nullptr && *next != '\0' && values.size() < 12u) {
				char* end = nullptr;
				const auto threshold = std::strtoull(next, &end, 10);
				if (end == next || threshold < 1u || threshold > 100000u) {
					break;
				}
				values.push_back(threshold);
				next = *end == ',' ? end + 1 : end;
				if (*end == '\0') {
					break;
				}
			}
			return values;
		}();
		if (m_present_capture_index >= thresholds.size() ||
		    index < thresholds[m_present_capture_index]) {
			return;
		}
		const std::string suffix = "-f" + std::to_string(thresholds[m_present_capture_index++]);
		const auto dot = capture_path.rfind(".png");
		capture_path.insert(dot == std::string::npos ? capture_path.size() : dot, suffix);
	} else {
		if (m_present_capture_done) {
			return;
		}
		uint64_t threshold = 600u;
		if (const char* requested = std::getenv("KYTY_TEST_CAPTURE_AFTER_FRAMES");
		    requested != nullptr && requested[0] != '\0') {
			const auto number = std::strtoull(requested, nullptr, 10);
			if (number >= 1u && number <= 100000u) {
				threshold = number;
			}
		}
		if (index < threshold) {
			return;
		}
		m_present_capture_done = true;
	}

	switch (m_format) {
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32: break;
		default:
			LOGF("Present frame capture: unsupported format=%d\n", static_cast<int>(m_format));
			return;
	}
	const auto width = m_extent.width;
	const auto height = m_extent.height;
	const uint64_t bytes = static_cast<uint64_t>(width) * height * 4u;
	if (width == 0 || height == 0 || width > 8192u || height > 8192u ||
	    bytes > 256u * 1024u * 1024u) {
		LOGF("Present frame capture: unsuitable extent=%ux%u\n", width, height);
		return;
	}
	EXIT_IF(m_present_capture_buffer != VK_NULL_HANDLE);
	auto& graphics = m_window.graphic_ctx;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = bytes;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	VkBuffer native_buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	if (vmaCreateBuffer(graphics.allocator, &buffer_info, &allocation_info,
	                    &native_buffer, &allocation, nullptr) != VK_SUCCESS) {
		LOGF("Present frame capture: failed to allocate staging buffer\n");
		return;
	}
	m_present_capture_buffer = native_buffer;
	m_present_capture_allocation = allocation;
	m_present_capture_path = std::move(capture_path);
	m_present_capture_extent = m_extent;
	m_present_capture_format = m_format;
	m_present_capture_frame = index;

	// The image is still acquired here. Read back the crop and all overlays in
	// the same submission, then restore the layout expected by vkQueuePresentKHR.
	vk::ImageMemoryBarrier2 to_copy {};
	to_copy.srcStageMask = vk::PipelineStageFlagBits2::eTransfer |
	                       vk::PipelineStageFlagBits2::eColorAttachmentOutput;
	to_copy.srcAccessMask = vk::AccessFlagBits2::eTransferWrite |
	                        vk::AccessFlagBits2::eColorAttachmentWrite;
	to_copy.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
	to_copy.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
	to_copy.oldLayout = vk::ImageLayout::ePresentSrcKHR;
	to_copy.newLayout = vk::ImageLayout::eTransferSrcOptimal;
	to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_copy.image = m_images[m_image_index];
	to_copy.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &to_copy;
	command.pipelineBarrier2(dependency);

	const vk::Buffer buffer = native_buffer;
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.imageExtent = {width, height, 1};
	command.copyImageToBuffer(m_images[m_image_index], vk::ImageLayout::eTransferSrcOptimal,
	                          buffer, 1, &copy);

	vk::BufferMemoryBarrier2 to_host {};
	to_host.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
	to_host.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	to_host.dstStageMask = vk::PipelineStageFlagBits2::eHost;
	to_host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.buffer = buffer;
	to_host.size = bytes;
	vk::ImageMemoryBarrier2 to_present = to_copy;
	to_present.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
	to_present.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	to_present.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	to_present.dstAccessMask = vk::AccessFlagBits2::eMemoryRead;
	to_present.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
	to_present.newLayout = vk::ImageLayout::ePresentSrcKHR;
	dependency.pImageMemoryBarriers = &to_present;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers = &to_host;
	command.pipelineBarrier2(dependency);
}

void Swapchain::FinishPresentCapture(CommandScheduler& scheduler, uint64_t tick) {
	if (m_present_capture_buffer == VK_NULL_HANDLE) {
		return;
	}
	scheduler.Wait(tick);
	auto& graphics = m_window.graphic_ctx;
	void* mapped = nullptr;
	const auto map_result = vmaMapMemory(graphics.allocator, m_present_capture_allocation, &mapped);
	if (map_result == VK_SUCCESS) {
		const auto invalidate_result = vmaInvalidateAllocation(
		    graphics.allocator, m_present_capture_allocation, 0, VK_WHOLE_SIZE);
		if (mapped != nullptr && invalidate_result == VK_SUCCESS) {
			// The window displays opaque RGB; do not let guest alpha hide the PNG.
			const int ok = WriteDiagnosticPng(m_present_capture_path,
			                                  m_present_capture_extent.width,
			                                  m_present_capture_extent.height,
			                                  m_present_capture_format,
			                                  static_cast<const uint8_t*>(mapped), true);
			LOGF("Present frame capture: file=%s extent=%ux%u format=%d success=%d frame=%" PRIu64
			     " host_ms=%" PRIu64 "\n", m_present_capture_path.c_str(),
			     m_present_capture_extent.width, m_present_capture_extent.height,
			     static_cast<int>(m_present_capture_format), ok, m_present_capture_frame,
			     static_cast<uint64_t>(SDL_GetTicks()));
		} else {
			LOGF("Present frame capture: could not invalidate host buffer\n");
		}
		vmaUnmapMemory(graphics.allocator, m_present_capture_allocation);
	} else {
		LOGF("Present frame capture: could not map host buffer\n");
	}
	ReleasePresentCapture();
}

void Swapchain::ReleasePresentCapture() {
	if (m_present_capture_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(m_window.graphic_ctx.allocator, m_present_capture_buffer,
		                 m_present_capture_allocation);
	}
	m_present_capture_buffer = VK_NULL_HANDLE;
	m_present_capture_allocation = nullptr;
	m_present_capture_path.clear();
}

uint64_t Swapchain::Submit(CommandScheduler& scheduler) {
	EXIT_IF(m_frame_index >= m_image_acquired.size() || m_image_index >= m_render_complete.size());
	SubmitInfo submit;
	submit.AddWait(m_image_acquired[m_frame_index], 1, vk::PipelineStageFlagBits::eTransfer);
	submit.AddSignal(m_render_complete[m_image_index]);
	return scheduler.Submit(submit);
}

Swapchain::Status Swapchain::Present() {
	EXIT_IF(m_image_index >= m_render_complete.size());
	const auto         ready = m_render_complete[m_image_index];
	vk::PresentInfoKHR present {};
	present.sType              = vk::StructureType::ePresentInfoKHR;
	present.swapchainCount     = 1;
	present.pSwapchains        = &m_handle;
	present.pImageIndices      = &m_image_index;
	present.pWaitSemaphores    = &ready;
	present.waitSemaphoreCount = 1;

	vk::Result result;
	{
		Common::LockGuard lock(m_window.graphic_ctx.queue_mutex);
		result = m_window.graphic_ctx.queue.presentKHR(&present);
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkQueuePresentKHR failed: %s\n", vk::to_string(result).c_str());
	}
	m_frame_index = (m_frame_index + 1u) % static_cast<uint32_t>(m_images.size());
	return Status::Success;
}

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {}

Presenter::~Presenter() = default;

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info) {
    KYTY_PROFILER_FUNCTION();
    EXIT_IF(buffer.IsInvalid());
    // Wait for a free presentation frame before taking the renderer lock:
    // frame acquisition can block on a GPU tick that needs the same mutex.
    auto initial_format = info.pixel_format;
    switch (initial_format) {
        case vk::Format::eR8G8B8A8Srgb: initial_format = vk::Format::eR8G8B8A8Unorm; break;
        case vk::Format::eB8G8R8A8Srgb: initial_format = vk::Format::eB8G8R8A8Unorm; break;
        default: break;
    }
    auto* frame = m_impl->frames.Acquire(
        {info.extent.width, info.extent.height}, initial_format);
    Common::LockGuard render_lock(m_impl->renderer.GetMutex());
    // Native PS VR2 reprojection normally fills a 4000x2040 panel buffer.
    // Without compositor hardware, allow an explicit desktop-only stereo
    // blit of the guest's observed 2000x2040 RGBA8 eye surfaces.
    if (Config::VrEnabled() && info.extent.width == 4000u &&
        info.extent.height == 2040u) {
        const char* stereo = std::getenv("KYTY_VR_STEREO_PREVIEW");
        if (stereo != nullptr && stereo[0] == '1' && stereo[1] == '\0') {
            auto& cache = m_impl->renderer.GetTextureCache();
            ImageId left_id {}, right_id {};
            if (cache.FindReprojectionEyePair(&left_id, &right_id)) {
                auto& left = cache.GetImage(left_id);
                auto& right = cache.GetImage(right_id);
                left.usage.video_out = true;
                right.usage.video_out = true;
                cache.UpdateImage(left_id);
                cache.UpdateImage(right_id);
                frame->Configure(m_impl->window.graphic_ctx,
                                 {4000u, 2040u}, initial_format);
                frame->BlitStereo(buffer, left, right);
                static std::atomic<uint32_t> logs {0};
                if (logs.fetch_add(1u, std::memory_order_relaxed) < 4u) {
                    LOGF("Hmd2: diagnostic stereo eye mirror left=%016" PRIx64
                         " right=%016" PRIx64 "\n",
                         left.info.data.address, right.info.data.address);
                }
                return *frame;
            }
        }
    }
    ImageInfo source = info;
    bool direct_source = false;
    ImageId candidate_image_id {};
    // Optional mono eye preview for desktops without PS VR2 hardware.
    // The observed Horizon eye textures are RGBA8 2000x2040.
    // This does not affect normal presentation without the environment flag.
    if (Config::VrEnabled() && info.extent.width == 3328u &&
        info.extent.height == 1664u) {
        if (const char* eye = std::getenv("KYTY_VR_PREVIEW_EYE");
            eye != nullptr && (eye[0] == '0' || eye[0] == '1') &&
            eye[1] == '\0') {
            direct_source = m_impl->renderer.GetTextureCache()
                .FindReprojectionCandidateByFormat(
                    56u, static_cast<uint32_t>(eye[0] - '0'),
                    &source, &candidate_image_id, 2000u, 2040u);
            static std::atomic<uint32_t> logs {0};
            if (logs.fetch_add(1u, std::memory_order_relaxed) < 4u) {
                LOGF("Hmd2: mono eye preview eye=%c found=%d gpu=%016" PRIx64 "\n",
                     eye[0], direct_source, source.data.address);
            }
        }
    }
    if (!direct_source && Config::VrEnabled() && info.extent.width == 3328 &&
        info.extent.height == 1664) {
        uint32_t series_id = 0u;
        if (!PreviewImageSeries().empty()) {
            series_id = PreviewImageSeries()[PreviewSeriesIndex(
                m_impl->window.loop.presented_frames.load(std::memory_order_relaxed))];
        }
        if (const char* candidate = std::getenv("KYTY_VR_PREVIEW_ID");
            series_id != 0u || (candidate != nullptr && candidate[0] != '\0')) {
            const auto id = series_id != 0u ? series_id : std::strtoul(candidate, nullptr, 10);            if (id <= 100000u) {
                                if (id >= 30000u && id < 40000u) {
                    const auto packed = static_cast<uint32_t>(id - 30000u);
                    direct_source = m_impl->renderer.GetTextureCache()
                        .FindReprojectionCandidateByFormat(
                            packed / 100u, packed % 100u, &source, &candidate_image_id);
                } else {
                    direct_source = m_impl->renderer.GetTextureCache()
                        .FindReprojectionCandidate(static_cast<uint32_t>(id), &source,
                                                  &candidate_image_id);
                }
                static std::atomic<uint32_t> logs {0};
                if (logs.fetch_add(1u, std::memory_order_relaxed) < 4u) {
                    LOGF("Hmd2: preview candidate id=%lu found=%d address=%016" PRIx64
                         " extent=%ux%u fmt=%u\n", id, direct_source,
                         source.data.address, source.extent.width, source.extent.height,
                         static_cast<uint32_t>(source.guest_format));
                }
            }
        }
    }
    auto frame_format = source.pixel_format;
    switch (frame_format) {
        case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
        case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
        default: break;
    }

    auto& image = m_impl->ResolveSurface(source,
                                         direct_source ? &candidate_image_id : nullptr);
    if (image.backing.format == vk::Format::eUndefined) {
        EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
    }
    frame->Configure(m_impl->window.graphic_ctx,
                     {image.backing.extent.width, image.backing.extent.height}, frame_format);
    frame->CopyFrom(buffer, image);
    return *frame;
}
Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	auto              format = m_impl->frames.GetFormat();
	auto*             frame  = m_impl->frames.Acquire({width, height}, format);
	Common::LockGuard render_lock(m_impl->renderer.GetMutex());
	frame->Configure(m_impl->window.graphic_ctx, {width, height}, format);
	vk::ClearColorValue clear {};
	clear.float32[3] = opaque ? 1.0f : 0.0f;
	if (producer != nullptr) {
		EXIT_IF(producer->IsInvalid());
		frame->Clear(*producer, clear);
	} else {
		auto& command = m_impl->present_scheduler.BeginCommand();
		frame->Clear(command, clear);
		frame->present_tick = m_impl->present_scheduler.Submit();
	}
	return *frame;
}

bool Presenter::PresentLastFrame() {
	Common::LockGuard lock(m_impl->present_mutex);
	if (m_impl->layers[0].frame == nullptr && m_impl->layers[1].frame == nullptr) {
		return false;
	}
	m_impl->Present();
	return true;
}

bool Presenter::IsGuestPaused() const noexcept {
	return m_impl->window.loop.paused.load(std::memory_order_acquire);
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
	const auto visual = GetSystemOverlayVisualState();
	return visual.active ||
	       visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

void Presenter::Present(Frame& frame) {
	const Layer layer {&frame, 0, false};
	Present(std::span(&layer, 1));
}

void Presenter::Present(std::span<const Layer> layers) {
	Common::LockGuard lock(m_impl->present_mutex);
	for (const auto& layer: layers) {
		EXIT_IF(layer.bus < 0 || layer.bus >= static_cast<int>(m_impl->layers.size()));
		m_impl->frames.ValidateForPresent(layer.frame);
		auto& previous = m_impl->layers[layer.bus];
		if (previous.frame != nullptr) {
			m_impl->frames.Release(previous.frame);
		}
		previous = layer;
	}
	m_impl->Present();
}

void Presenter::ClearLayer(int bus) {
	if (static_cast<size_t>(bus) >= m_impl->layers.size()) {
		return;
	}
	Common::LockGuard lock(m_impl->present_mutex);
	auto& layer = m_impl->layers[bus];
	if (layer.frame != nullptr) {
		m_impl->frames.Release(layer.frame);
		layer = {};
	}
}

void Presenter::Impl::CaptureFrameIfRequested(Presenter::Frame* frame) {
    const char* filename = std::getenv("KYTY_TEST_FRAME_CAPTURE");
    if (filename == nullptr || filename[0] == '\0' || frame == nullptr) {
        return;
    }
    const auto index = window.loop.presented_frames.load(std::memory_order_relaxed);
    const auto& series = PreviewImageSeries();
    static bool captured = false;
    static int last_series_capture = -1;
    std::string capture_path(filename);
    if (!series.empty()) {
        if (index < kPreviewSeriesStart + 40u) {
            return;
        }
        const auto slot = PreviewSeriesIndex(index);
        if (slot == static_cast<size_t>(last_series_capture) ||
            index < kPreviewSeriesStart + slot * kPreviewSeriesFrames + 40u) {
            return;
        }
        last_series_capture = static_cast<int>(slot);
        const std::string suffix = "-id" + std::to_string(series[slot]);
        const auto dot = capture_path.rfind(".png");
        capture_path.insert(dot == std::string::npos ? capture_path.size() : dot, suffix);
    } else if (const char* points = std::getenv("KYTY_TEST_CAPTURE_POINTS");
               points != nullptr && points[0] != '\0') {
        static const std::vector<uint64_t> thresholds = [] {
            std::vector<uint64_t> values;
            const char* next = std::getenv("KYTY_TEST_CAPTURE_POINTS");
            while (next != nullptr && *next != '\0' && values.size() < 12u) {
                char* end = nullptr;
                const auto threshold = std::strtoull(next, &end, 10);
                if (end == next || threshold < 1u || threshold > 100000u) {
                    break;
                }
                values.push_back(threshold);
                next = *end == ',' ? end + 1 : end;
                if (*end == '\0') {
                    break;
                }
            }
            return values;
        }();
        static size_t capture_index = 0;
        if (capture_index >= thresholds.size() ||
            index < thresholds[capture_index]) {
            return;
        }
        const std::string suffix = "-f" + std::to_string(thresholds[capture_index]);
        const auto dot = capture_path.rfind(".png");
        capture_path.insert(dot == std::string::npos ? capture_path.size() : dot,
                            suffix);
        ++capture_index;
    } else {
        if (captured) {
            return;
        }
        uint64_t threshold = 600u;
        if (const char* requested = std::getenv("KYTY_TEST_CAPTURE_AFTER_FRAMES");
            requested != nullptr && requested[0] != '\0') {
            const auto number = std::strtoull(requested, nullptr, 10);
            if (number >= 1u && number <= 100000u) {
                threshold = number;
            }
        }
        if (index < threshold) {
            return;
        }
        captured = true;
    }

    // One synchronous GPU readback per selected VR diagnostic source.
    const auto width = frame->image.extent.width;
    const auto height = frame->image.extent.height;
    if (width == 0 || height == 0 || width > 8192 || height > 8192 ||
        frame->image.layers != 1 ||
        !(frame->image.usage & vk::ImageUsageFlagBits::eTransferSrc)) {
        LOGF("Hidden frame capture: unsuitable source extent=%ux%u\n", width, height);
        return;
    }
    const uint64_t bytes = static_cast<uint64_t>(width) * height * 4u;
    if (bytes > 256u * 1024u * 1024u) {
        return;
    }
    auto& graphics = window.graphic_ctx;
    VkBufferCreateInfo buffer_info {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = bytes;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_info {};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkBuffer native_buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    if (vmaCreateBuffer(graphics.allocator, &buffer_info, &allocation_info,
                        &native_buffer, &allocation, nullptr) != VK_SUCCESS) {
        LOGF("Hidden frame capture: failed to allocate staging buffer\n");
        return;
    }
    const vk::Buffer buffer = native_buffer;

    auto& command_buffer = present_scheduler.BeginCommand();
    const auto command = command_buffer.Handle();
    frame->Transit(command, vk::ImageLayout::eTransferSrcOptimal,
                   vk::AccessFlagBits2::eTransferRead);
    vk::BufferImageCopy copy {};
    copy.bufferOffset = 0;
    copy.bufferRowLength = 0;
    copy.bufferImageHeight = 0;
    copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copy.imageOffset = {0, 0, 0};
    copy.imageExtent = {width, height, 1};
    command.copyImageToBuffer(frame->image.image,
                              vk::ImageLayout::eTransferSrcOptimal,
                              buffer, 1, &copy);

    vk::BufferMemoryBarrier2 barrier {};
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eHost;
    barrier.dstAccessMask = vk::AccessFlagBits2::eHostRead;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = bytes;
    vk::DependencyInfo dependency {};
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;
    command.pipelineBarrier2(dependency);

    const auto tick = present_scheduler.Submit();
    present_scheduler.Wait(tick);

    void* mapped = nullptr;
    const auto map_result = vmaMapMemory(graphics.allocator, allocation, &mapped);
    if (map_result == VK_SUCCESS && mapped != nullptr) {
        vmaInvalidateAllocation(graphics.allocator, allocation, 0, VK_WHOLE_SIZE);
        const auto* raw = static_cast<const uint8_t*>(mapped);
        const auto fmt = frame->image.format;
        const int ok = WriteDiagnosticPng(capture_path, width, height, fmt, raw);
        LOGF("Hidden frame capture: file=%s extent=%ux%u format=%d success=%d frame=%" PRIu64 " host_ms=%" PRIu64 "\n",
             capture_path.c_str(), width, height, static_cast<int>(fmt), ok,
             index, static_cast<uint64_t>(SDL_GetTicks()));
        vmaUnmapMemory(graphics.allocator, allocation);
    } else {
        LOGF("Hidden frame capture: could not map host buffer\n");
    }
    vmaDestroyBuffer(graphics.allocator, native_buffer, allocation);
}
void Presenter::Impl::Present() {
	KYTY_PROFILER_FUNCTION();

	const auto overlay_visual = GetSystemOverlayVisualState();
	// Some window systems keep presenting an old swapchain after a resize.
	if (swapchain.NeedsResize()) {
		RecoverSwapchain(Swapchain::Status::Recreate);
	}
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		auto status = swapchain.AcquireNextImage();
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			continue;
		}
		{
			Common::LockGuard render_lock(renderer.GetMutex());
			auto&             command = present_scheduler.BeginCommand();
			const bool        draw_system_overlay =
			    overlay_visual.active && swapchain.PrepareSystemOverlay();
			swapchain.RecordPresentCommands(command, layers[0].frame, layers[1],
			                                draw_system_overlay);
			const auto tick = swapchain.Submit(present_scheduler);
			swapchain.FinishPresentCapture(present_scheduler, tick);
			CaptureFrameIfRequested(layers[0].frame);
			for (const auto& layer: layers) {
				if (layer.frame != nullptr) {
					layer.frame->present_tick = tick;
				}
			}
		}
		status = swapchain.Present();
		if (status != Swapchain::Status::Success) {
			RecoverSwapchain(status);
			continue;
		}

		presented_overlay_revision.store(overlay_visual.revision, std::memory_order_release);
		window.loop.presented_frames.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
