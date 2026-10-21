#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps({front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		                            {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	dynamic_state.SetColorWriteEnables(std::span {enable}.first(rendering.num_color_attachments));
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || meta_clear;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			EXIT("depth attachment feedback loop is not supported by the host\n");
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh_regs.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh_regs.target_output_mode[index] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	state.ps_input_info = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	RefreshShaders(buffer, draw, color_output_mask, state);
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.location;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

namespace {

struct PanelUpstreamOptions {
	bool enabled = false;
	uint64_t parent_ps_hash = 0;
	std::array<uint32_t, 2> capture_frames {2048u, 3072u};
};

const PanelUpstreamOptions& GetPanelUpstreamOptions() {
	static const PanelUpstreamOptions options = [] {
		PanelUpstreamOptions result;
		const char* enabled = std::getenv("KYTY_VR_PANEL_UPSTREAM_TRACE");
		if (enabled == nullptr || std::strcmp(enabled, "1") != 0) return result;
		const auto disabled = [&](const char* reason) {
			LOGF("PanelProducer upstream disabled: reason=%s existing-producer-diagnostic-unchanged\n", reason);
			return result;
		};
		const char* hash = std::getenv("KYTY_VR_PANEL_UPSTREAM_PARENT_PS_HASH");
		if (hash == nullptr || std::strlen(hash) != 16u) return disabled("parent-hash-must-be-16-hex-digits");
		for (uint32_t i = 0; i < 16u; ++i) {
			const char c = hash[i];
			const uint32_t digit = c >= '0' && c <= '9' ? c - '0'
			    : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 16u;
			if (digit >= 16u) return disabled("parent-hash-must-be-16-hex-digits");
			result.parent_ps_hash = (result.parent_ps_hash << 4u) | digit;
		}
		const auto read_frame = [](const char* name, uint32_t& frame) {
			const char* value = std::getenv(name);
			if (value == nullptr) return true;
			if (*value == '\0') return false;
			uint32_t parsed = 0;
			for (; *value != '\0'; ++value) {
				if (*value < '0' || *value > '9') return false;
				parsed = parsed * 10u + static_cast<uint32_t>(*value - '0');
				if (parsed > 4096u) return false;
			}
			frame = parsed;
			return parsed != 0;
		};
		if (!read_frame("KYTY_VR_PANEL_UPSTREAM_CAPTURE_FIRST_FRAME", result.capture_frames[0]) ||
		    !read_frame("KYTY_VR_PANEL_UPSTREAM_CAPTURE_SECOND_FRAME", result.capture_frames[1]) ||
		    result.capture_frames[0] >= result.capture_frames[1]) return disabled("capture-frames-require-0<first<second<=4096");
		result.enabled = true;
		LOGF("PanelProducer upstream enabled: depth=1 parent_ps=%016" PRIx64
		     " primary_slots=0:8 upstream_slots=8:4 recursive_promotion=0 capture_frames=%u,%u "
		     "capture_budget=reuse-producer-2sets-64MiB-per-set-128MiB-total discovery_windows=8x3frames "
		     "uncovered=pre-discovery-writes,cache-color-materialization,cpu-uploads,depth-transfer-shortcuts,non-front-dynamic-mip-views\n",
		     result.parent_ps_hash, result.capture_frames[0], result.capture_frames[1]);
		return result;
	}();
	return options;
}

struct PanelUpstreamCaptureFilter {
	// Keep logical selection inside the existing bounded per-draw image list.
	static constexpr uint32_t MaxImageIndex = 15u;
	bool enabled = true;
	bool exact = false;
	uint64_t ps_hash = 0;
	uint32_t image_index = 0;
};

constexpr uint64_t PanelMultiplierPrimaryPs = 0x83b1c2d863c50cf3ull;
constexpr uint64_t PanelMultiplierVertexShader = 0x492fcfff7e74a877ull;

const PanelUpstreamCaptureFilter& GetPanelUpstreamCaptureFilter() {
	static const PanelUpstreamCaptureFilter options = [] {
		PanelUpstreamCaptureFilter result;
		const char* hash = std::getenv("KYTY_VR_PANEL_UPSTREAM_CAPTURE_PS_HASH");
		const char* index = std::getenv("KYTY_VR_PANEL_UPSTREAM_CAPTURE_IMAGE_INDEX");
		if (hash == nullptr && index == nullptr) return result;
		const auto disabled = [&](const char* reason) {
			result.enabled = false;
			LOGF("PanelProducer capture disabled: reason=%s ancestry-tracing-unchanged\n", reason);
			return result;
		};
		if (hash == nullptr || index == nullptr) return disabled("capture-filter-requires-both-ps-hash-and-image-index");
		if (std::strlen(hash) != 16u) return disabled("capture-ps-hash-must-be-16-hex-digits");
		for (uint32_t i = 0; i < 16u; ++i) {
			const char c = hash[i];
			const uint32_t digit = c >= '0' && c <= '9' ? c - '0'
			    : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 16u;
			if (digit >= 16u) return disabled("capture-ps-hash-must-be-16-hex-digits");
			result.ps_hash = (result.ps_hash << 4u) | digit;
		}
		if (*index == '\0') return disabled("capture-image-index-must-be-decimal-0-through-15");
		for (; *index != '\0'; ++index) {
			if (*index < '0' || *index > '9') return disabled("capture-image-index-must-be-decimal-0-through-15");
			const uint32_t digit = static_cast<uint32_t>(*index - '0');
			if (result.image_index > PanelUpstreamCaptureFilter::MaxImageIndex / 10u ||
			    (result.image_index == PanelUpstreamCaptureFilter::MaxImageIndex / 10u &&
			     digit > PanelUpstreamCaptureFilter::MaxImageIndex % 10u))
				return disabled("capture-image-index-must-be-decimal-0-through-15");
			result.image_index = result.image_index * 10u + digit;
		}
		result.exact = true;
		LOGF("PanelProducer capture filter: mode=exact-pixel-input ps=%016" PRIx64
		     " logical_image=%u index_scope=program.info.images image_index_max=%u ancestry-tracing-unchanged\n",
		     result.ps_hash, result.image_index, PanelUpstreamCaptureFilter::MaxImageIndex);
		return result;
	}();
	return options;
}

bool PanelVertexMultiplierCaptureEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_VR_PANEL_CAPTURE_VERTEX_MULTIPLIER");
		if (value == nullptr || std::strcmp(value, "1") != 0) return false;
		const auto& filter = GetPanelUpstreamCaptureFilter();
		if (!filter.enabled || !filter.exact || filter.ps_hash != PanelMultiplierPrimaryPs ||
		    filter.image_index != 3u) {
			LOGF("PanelProducer vertex multiplier capture disabled: reason=requires-exact-83b1c2d863c50cf3-input3-filter\n");
			return false;
		}
		LOGF("PanelProducer vertex multiplier capture enabled: primary_ps=%016" PRIx64
		     " primary_image=3 vertex_shader=%016" PRIx64
		     " vertex_image=0 primary-must-queue-first=1 shared-input-cap=2\n",
		     PanelMultiplierPrimaryPs, PanelMultiplierVertexShader);
		return true;
	}();
	return enabled;
}

template <typename Handle>
uint64_t PanelNativeHandle(Handle handle) {
	using Native = typename Handle::CType;
	if constexpr (std::is_pointer_v<Native>) {
		return reinterpret_cast<uintptr_t>(static_cast<Native>(handle));
	} else {
		return static_cast<uint64_t>(static_cast<Native>(handle));
	}
}

uint32_t PanelCaptureTexelBytes(vk::Format format) {
	// Raw captures retain the native encoding; no color conversion or format guessing.
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8Uint: return 1;
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16Sfloat: return 2;
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eR8G8B8A8Uint:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eE5B9G9R9UfloatPack32:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32Uint: return 4;
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32G32Sfloat: return 8;
		case vk::Format::eR32G32B32A32Sfloat: return 16;
		default: return 0;
	}
}

bool PanelImagesOverlap(const Image& a, const Image& b) {
	if (a.backing.image == b.backing.image) return true;
	for (const auto left: {a.info.data, a.info.stencil, a.info.metadata.range}) {
		for (const auto right: {b.info.data, b.info.stencil, b.info.metadata.range}) {
			if (!left.Empty() && !right.Empty() &&
			    ImageRangeOverlaps(left.address, left.size, right.address, right.size)) return true;
		}
	}
	return false;
}

struct PanelGpuScalarCaptureOptions {
	bool enabled = false;
	bool hdr_uv = false;
	bool composite = false;
	std::string prefix;
	std::array<uint32_t, 2> milestones {2048u, 3072u};
};

const PanelGpuScalarCaptureOptions& GetPanelGpuScalarCaptureOptions() {
	static const PanelGpuScalarCaptureOptions options = [] {
		PanelGpuScalarCaptureOptions result;
		const char* value = std::getenv("KYTY_VR_PANEL_GPU_SCALAR_CAPTURE");
		const bool tone_enabled = value != nullptr && std::strcmp(value, "1") == 0;
		const char* composite = std::getenv("KYTY_VR_PANEL_GPU_COMPOSITE_CAPTURE");
		const bool composite_enabled = composite != nullptr && std::strcmp(composite, "1") == 0;
		if (!tone_enabled && !composite_enabled) return result;
		const auto disabled = [&](const char* reason) {
			LOGF("PanelProducer GPU scalar capture disabled: reason=%s existing-diagnostics-unchanged\n", reason);
			return result;
		};
		if (tone_enabled && composite_enabled)
			return disabled("mutually-exclusive-tone-and-composite-modes");
		const char* hdr_uv = std::getenv("KYTY_VR_PANEL_GPU_SCALAR_CAPTURE_HDR_UV");
		const bool hdr_uv_enabled = hdr_uv != nullptr && std::strcmp(hdr_uv, "1") == 0;
		if (composite_enabled && hdr_uv_enabled)
			return disabled("composite-mode-rejects-hdr-uv-companion");
		const char* producer = std::getenv("KYTY_VR_PANEL_PRODUCER_TRACE");
		if (producer == nullptr || std::strcmp(producer, "1") != 0)
			return disabled("requires-producer-trace");
		const auto& upstream = GetPanelUpstreamOptions();
		if (!upstream.enabled) return disabled("requires-valid-upstream-trace");
		const char* prefix = std::getenv("KYTY_VR_PANEL_GPU_SCALAR_CAPTURE_PREFIX");
		if (prefix == nullptr || *prefix == '\0' || std::strlen(prefix) > 2048u)
			return disabled("requires-nonempty-prefix-at-most-2048-bytes");
		result.prefix = prefix;
		result.milestones = upstream.capture_frames;
		result.hdr_uv = hdr_uv_enabled;
		result.composite = composite_enabled;
		result.enabled = true;
		if (result.composite) {
			LOGF("PanelProducer GPU composite capture enabled: ps=8f4f493f4d30416d resource=3 binding=3 "
			     "byte_offset=0 bytes=28 timing=before-draw capture_frames=%u,%u max_captures=2 "
			     "max_raw_bytes=152 max_decisions=64 cutoff_frame=4096 guest_reads=0 image-budget-unchanged=1 "
			     "companion_recipe=8f4f493f4d30416d:0:0:48 primary-first=1\n",
			     result.milestones[0], result.milestones[1]);
		} else {
			LOGF("PanelProducer GPU scalar capture enabled: ps=83b1c2d863c50cf3 resource=1 binding=1 "
			     "byte_offset=2168 bytes=4 timing=before-draw capture_frames=%u,%u max_captures=2 "
			     "max_raw_bytes=%u max_decisions=64 cutoff_frame=4096 guest_reads=0 image-budget-unchanged=1 "
			     "hdr_uv_companion=%d companion_recipe=83b1c2d863c50cf3:0:0:48 primary-first=1\n",
			     result.milestones[0], result.milestones[1], result.hdr_uv ? 104u : 8u, result.hdr_uv);
		}
		return result;
	}();
	return options;
}

bool PanelScalarRangeFits(uint64_t offset, uint64_t size, uint64_t capacity) {
	return offset <= capacity && size <= capacity - offset;
}

const Buffer* ResolvePanelScalarNativeOwner(BufferCache& cache,
	                                        const vk::DescriptorBufferInfo& native,
	                                        uint64_t address, uint64_t size, bool& stream) {
	stream = false;
	if (address == 0 || size == 0 || !GuestRange {address, size}.Valid() ||
	    native.buffer == nullptr || native.range != size || native.range == VK_WHOLE_SIZE)
		return nullptr;
	const auto& upload = cache.GetUtilityBuffer(MemoryUsage::Stream);
	const Buffer* owner = nullptr;
	if (native.buffer == upload.Handle()) {
		if (upload.Usage() != MemoryUsage::Stream) return nullptr;
		owner = &upload;
		stream = true;
	} else {
		owner = cache.FindRegisteredBuffer(address, size);
		if (owner == nullptr || owner->Usage() != MemoryUsage::DeviceLocal ||
		    owner->Handle() != native.buffer || owner->Offset(address) != native.offset)
			return nullptr;
	}
	// Both accepted allocation routes use AllFlags, including TransferSrc. Do not
	// infer transfer usage or ownership for an arbitrary native descriptor handle.
	return !owner->is_deleted && PanelScalarRangeFits(native.offset, native.range, owner->Size())
	           ? owner : nullptr;
}

bool PanelNativeScalarTraceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_VR_PANEL_NATIVE_SCALAR_TRACE");
		const bool result = value != nullptr && std::strcmp(value, "1") == 0;
		if (result) LOGF("PanelProducer native scalar trace: mapped-stream-only recipe_fields=ps:resource:byte_offset:bytes "
		                 "recipes=7b2dcbdf832f0791:0:0:32,8f4f493f4d30416d:3:0:4,83b1c2d863c50cf3:1:2168:4 "
		                 "window=existing-graphics-native-image-writes "
		                 "gpu_completed=0 guest_reads=0\n");
		return result;
	}();
	return enabled;
}

void AppendPanelNativeScalars(std::string& record, uint32_t call, const CommandBuffer& buffer,
                              std::span<const RenderColorInfo> colors,
                              std::span<const ImageInfo* const> attachment_infos, bool invalid_attachment,
                              std::span<PreparedBindings* const> stages) {
	// Called only after the actual draw, in an existing selected producer window. The
	// source is the final native descriptor, never its historical guest backing address.
	if (!GuestGpu::IsGpuThread()) {
		record += fmt::format("PanelProducer native-scalar call={} state=skipped reason=not-gpu-thread\n", call);
		return;
	}
	auto& context = buffer.GetContext();
	const auto& stream = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const std::span<const uint8_t> mapped = stream.Mapped();
	for (uint32_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
		const auto* prepared = stages[stage_index];
		if (prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime) {
			record += fmt::format("PanelProducer native-scalar call={} stage={} state=skipped reason=invalid-stage\n",
			                      call, stage_index);
			continue;
		}
		const auto& program = *prepared->runtime->program;
		if (program.stage != ShaderType::Pixel) continue;
		const bool uv = program.shader_hash == 0x7b2dcbdf832f0791ull;
		const bool pre_lut = program.shader_hash == 0x83b1c2d863c50cf3ull;
		if (!uv && !pre_lut && program.shader_hash != 0x8f4f493f4d30416dull) continue;
		const uint32_t resource_index = uv ? 0u : pre_lut ? 1u : 3u;
		const uint32_t expected_pc = uv ? 0x8u : pre_lut ? 0x40u : 0x1d0u;
		const uint32_t declared_bytes = uv ? 32u : pre_lut ? 4384u : 48u;
		const uint32_t selected_offset = pre_lut ? 2168u : 0u;
		const uint32_t selected_bytes = uv ? 32u : 4u;
		const char* role = uv ? "uv-transform-and-bounds" : pre_lut ? "pre-LUT-common-RGB-multiplier" : "scene-background-mix";
		const auto& snapshot = *prepared->runtime->resources;
		ShaderBufferResource descriptor {};
		const PreparedBindings::BufferSource* source = nullptr;
		const vk::DescriptorBufferInfo* native = nullptr;
		uint32_t binding_index = UINT32_MAX;
		uint32_t adjustment = UINT32_MAX;
		uint64_t selected_native_offset = 0;
		bool selected_range_valid = false;
		std::array<uint32_t, 8> values {};
		const char* reason = [&]() -> const char* {
			if (resource_index >= program.info.buffers.size() || resource_index >= snapshot.buffers.size())
				return "missing-recipe-resource";
			const auto& resource = program.info.buffers[resource_index];
			binding_index = resource.descriptor_index;
			if (binding_index < prepared->buffer_sources.size()) source = &prepared->buffer_sources[binding_index];
			if (binding_index < prepared->buffers.size()) native = &prepared->buffers[binding_index];
			const auto& words = snapshot.buffers[resource_index];
			if (words.dword_count != 4u) return "descriptor-word-count";
			std::memcpy(descriptor.fields, words.dwords.data(), sizeof(descriptor.fields));
			if (binding_index != resource_index || resource.first_use_pc != expected_pc ||
			    !resource.read || !resource.scalar || resource.written || resource.atomic || resource.formatted ||
			    resource.image_alias != ShaderRecompiler::IR::BufferResource::NoImageAlias ||
			    resource.indirect_root != ShaderRecompiler::IR::BufferResource::NoIndirectBuffer ||
			    !resource.indirect_resources.empty() || !program.info.user_data_registers.empty())
				return "recipe-mapping-or-access";
			if (source == nullptr || native == nullptr) return "missing-native-binding";
			const uint64_t offset_word = uint64_t {program.bindings.memory_offset_dword} + binding_index / 4u;
			if (offset_word >= prepared->shader_data.size()) return "missing-packed-offset";
			adjustment = (prepared->shader_data[static_cast<size_t>(offset_word)] >> ((binding_index % 4u) * 8u)) & 255u;
			if (adjustment != 0u) return "nonzero-packed-offset";
			// These emitted recipes were proved with stride16, 2/3/0x112 records and
			// this exact non-address descriptor encoding. Reject changed variants.
			if ((descriptor.fields[1] & 0xffff0000u) != 0x00100000u || descriptor.fields[3] != 0x0004dfacu ||
			    descriptor.GetSize() != declared_bytes || source->address != descriptor.Base48() ||
			    source->size != declared_bytes || !GuestRange {source->address, source->size}.Valid())
				return "descriptor-or-resolved-range";
			if (native->buffer == nullptr || native->range == VK_WHOLE_SIZE || native->range != declared_bytes)
				return "native-range";
			if (native->buffer != stream.Handle()) return "not-mapped-stream";
			if (stream.Usage() != MemoryUsage::Stream || mapped.empty()) return "stream-not-mapped";
			const auto fits = [](uint64_t offset, uint64_t size, uint64_t capacity) {
				return offset <= capacity && size <= capacity - offset;
			};
			if (!fits(native->offset, native->range, stream.Size()) ||
			    !fits(native->offset, native->range, mapped.size()))
				return "mapped-extent";
			// Validate by subtraction before adding the proved logical byte offset.
			if (!fits(selected_offset, selected_bytes, native->range)) return "selected-native-extent";
			selected_native_offset = native->offset + selected_offset;
			if (!fits(selected_native_offset, selected_bytes, mapped.size())) return "selected-mapped-extent";
			selected_range_valid = true;
			for (const auto* other: stages) {
				if (other == nullptr || other->runtime == nullptr || !*other->runtime) return "invalid-bound-stage";
				const auto& other_program = *other->runtime->program;
				if (other_program.has_address_writes ||
				    std::any_of(other_program.info.buffers.begin(), other_program.info.buffers.end(),
				                [](const auto& r) { return r.written || r.atomic; })) return "bound-buffer-or-address-write";
				if (other->images.size() != other_program.info.images.size() ||
				    std::any_of(other_program.info.images.begin(), other_program.info.images.end(),
				                [](const auto& r) { return r.written || r.atomic || r.atomic64; }) ||
				    std::any_of(other->images.begin(), other->images.end(),
				                [](const auto& b) { return b.desc.type == TextureCache::BindingType::Storage; }))
					return "bound-storage-image-or-unknown-write";
			}
			const auto check_info = [&](const ImageInfo& info) -> const char* {
				for (const auto range: {info.data, info.stencil, info.metadata.range}) {
					if (range.Empty()) continue;
					if (!range.Valid()) return "invalid-attachment-range";
					if (ImageRangeOverlaps(source->address, source->size, range.address, range.size))
						return "attachment-overlap";
				}
				return nullptr;
			};
			if (invalid_attachment) return "unresolved-attachment";
			for (const auto* info: attachment_infos) {
				if (info == nullptr) return "unresolved-attachment";
				if (const auto* why = check_info(*info)) return why;
			}
			for (const auto& color: colors)
				if (const auto* why = check_info(color.desc.info)) return why;
			// The production Commit already flushed this host-written reservation and
			// protected its lifetime. Copy now; never keep a mapped pointer for later.
			std::memcpy(values.data(), mapped.data() + static_cast<size_t>(selected_native_offset), selected_bytes);
			return nullptr;
		}();
		record += fmt::format("PanelProducer native-scalar call={} stage={} hash={:016x} resource={} binding={} "
		    "state={} reason={} origin={} gpu_completed=0 sharp={:08x},{:08x},{:08x},{:08x} "
		    "guest={:016x} declared={} resolved={:016x}:{} native={:016x} offset={} range={} "
		    "stream={:016x} stream_size={} mapped_size={} coherent={} adjustment={} selected_offset={} selected_bytes={} "
		    "selected_native_offset={} selected_range_valid={} role={}",
		    call, stage_index, program.shader_hash, resource_index, binding_index,
		    reason == nullptr ? "captured" : "skipped", reason == nullptr ? "ok" : reason,
		    reason == nullptr ? "mapped-stream-upload" : "none", descriptor.fields[0], descriptor.fields[1],
		    descriptor.fields[2], descriptor.fields[3], descriptor.Base48(), descriptor.GetSize(),
		    source != nullptr ? source->address : 0, source != nullptr ? source->size : 0,
		    native != nullptr ? PanelNativeHandle(native->buffer) : 0, native != nullptr ? native->offset : 0,
		    native != nullptr ? native->range : 0, PanelNativeHandle(stream.Handle()), stream.Size(), mapped.size(),
		    stream.IsCoherent(), adjustment, selected_offset, selected_bytes, selected_native_offset, selected_range_valid, role);
		if (reason == nullptr) {
			record += " dwords=";
			for (uint32_t i = 0; i < selected_bytes / sizeof(uint32_t); ++i) record += fmt::format("{:08x},", values[i]);
			record += " floats=";
			for (uint32_t i = 0; i < selected_bytes / sizeof(uint32_t); ++i) record += fmt::format("{},", std::bit_cast<float>(values[i]));
		}
		record += "\n";
	}
}

struct PanelInputReadback {
	VmaAllocator allocator = nullptr;
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
	~PanelInputReadback() {
		if (mapped != nullptr) vmaUnmapMemory(allocator, allocation);
		if (buffer != VK_NULL_HANDLE) vmaDestroyBuffer(allocator, buffer, allocation);
	}
};

bool QueuePanelInputCapture(CommandBuffer& command, Image& image, uint32_t bytes,
	                            const std::string& path, std::string metadata, const char* label) {
	auto readback = std::make_unique<PanelInputReadback>();
	readback->allocator = command.GetGraphics().allocator;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = bytes;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	const auto result = vmaCreateBuffer(readback->allocator, &buffer_info, &allocation_info,
	                                    &readback->buffer, &readback->allocation, nullptr);
	if (result != VK_SUCCESS) {
		LOGF("%s capture allocation failed: result=%d bytes=%u\n", label, static_cast<int>(result), bytes);
		return false;
	}
	// Eligibility requires one native mip/array layer and no split state. Restore the exact
	// stage/access/layout with GetBarriers, since Transit derives a broader stage mask.
	const auto previous = image.backing.state;
	const vk::BufferImageCopy copy {0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	                               {0, 0, 0}, image.backing.extent};
	image.Download(std::span {&copy, 1}, readback->buffer, 0, bytes);
	vk::BufferMemoryBarrier2 host_barrier {};
	host_barrier.srcStageMask = vk::PipelineStageFlagBits2::eCopy;
	host_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	host_barrier.dstStageMask = vk::PipelineStageFlagBits2::eHost;
	host_barrier.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host_barrier.buffer = readback->buffer;
	host_barrier.offset = 0;
	host_barrier.size = bytes;
	const auto restore = image.GetBarriers(previous.layout, previous.access_mask,
	                                       previous.pl_stage, {});
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers = &host_barrier;
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(restore.size());
	dependency.pImageMemoryBarriers = restore.data();
	command.Handle().pipelineBarrier2(dependency);
	auto& scheduler = command.GetContext().GetCommandScheduler();
	const auto tick = scheduler.CurrentTick();
	const uint64_t row_bytes = static_cast<uint64_t>(image.backing.extent.width) *
	                           PanelCaptureTexelBytes(image.backing.format);
	const uint64_t slice_bytes = row_bytes * image.backing.extent.height;
	metadata += fmt::format("capture file={} bytes={} row_bytes={} native_format={} ({}) "
	                        "extent={}x{} tick={} layout={} access={:x} stage={:x} "
	                        "depth={} slice_bytes={} order=z-y-x native_type={}\n",
	                        path, bytes, row_bytes,
	                        static_cast<uint32_t>(image.backing.format), vk::to_string(image.backing.format),
	                        image.backing.extent.width, image.backing.extent.height, tick,
	                        static_cast<uint32_t>(previous.layout),
	                        static_cast<uint64_t>(previous.access_mask),
	                        static_cast<uint64_t>(previous.pl_stage), image.backing.extent.depth,
	                        slice_bytes, static_cast<uint32_t>(image.backing.image_type));
	// The normal deferred queue retains the dedicated allocation until this tick is
	// complete. Shutdown submits and drains that queue before the allocator is destroyed.
	scheduler.DeferOperation([readback = std::move(readback), path, metadata = std::move(metadata), bytes,
	                         capture_label = std::string(label)]() {
		uint32_t written = 0;
		bool saved = false;
		if (vmaMapMemory(readback->allocator, readback->allocation, &readback->mapped) == VK_SUCCESS &&
		    readback->mapped != nullptr &&
		    vmaInvalidateAllocation(readback->allocator, readback->allocation, 0, bytes) == VK_SUCCESS) {
			Common::File raw;
			if (raw.Create(path)) {
				raw.Write(readback->mapped, bytes, &written);
				saved = written == bytes && raw.Flush();
			}
		}
		Common::File info;
		bool metadata_saved = false;
		if (info.Create(path + ".txt")) {
			const auto text = metadata + fmt::format("saved={} written={}\n", saved, written);
			uint32_t count = 0;
			info.Write(text.data(), static_cast<uint32_t>(text.size()), &count);
			metadata_saved = count == text.size() && info.Flush();
		}
		LOGF("%s capture complete: file=%s bytes=%u saved=%d metadata=%d\n",
		     capture_label.c_str(), path.c_str(), bytes, saved, metadata_saved);
	});
	LOGF("%s capture queued: file=%s bytes=%u tick=%" PRIu64 "\n", label, path.c_str(), bytes, tick);
	return true;
}

} // namespace

const RenderExecutor::PanelGpuScalarRecipe* RenderExecutor::GetPanelGpuScalarRecipe(PanelGpuScalarKind kind) {
	static constexpr PanelGpuScalarRecipe common_rgb {0x83b1c2d863c50cf3ull, 1u, 0x40u, 4384u, 2168u, 4u};
	static constexpr PanelGpuScalarRecipe hdr_uv {0x83b1c2d863c50cf3ull, 0u, 0x10u, 128u, 0u, 48u};
	static constexpr PanelGpuScalarRecipe composite_controls {0x8f4f493f4d30416dull, 3u, 0x1d0u, 48u, 0u, 28u};
	static constexpr PanelGpuScalarRecipe composite_uv {0x8f4f493f4d30416dull, 0u, 0x10u, 48u, 0u, 48u};
	switch (kind) {
	case PanelGpuScalarKind::CommonRgb: return &common_rgb;
	case PanelGpuScalarKind::HdrUv: return &hdr_uv;
	case PanelGpuScalarKind::CompositeControls: return &composite_controls;
	case PanelGpuScalarKind::CompositeUv: return &composite_uv;
	default: return nullptr;
	}
}

const char* RenderExecutor::PlanPanelGpuScalarCapture(
	const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	const RenderState& rendering, std::span<PreparedBindings* const> stages,
	PanelGpuScalarPlan& plan) const {
	return PlanPanelGpuCapture(PanelGpuScalarKind::CommonRgb, buffer, colors, depth, rendering, stages, plan);
}

const char* RenderExecutor::PlanPanelGpuHdrUvCapture(
	const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	const RenderState& rendering, std::span<PreparedBindings* const> stages,
	PanelGpuScalarPlan& plan) const {
	return PlanPanelGpuCapture(PanelGpuScalarKind::HdrUv, buffer, colors, depth, rendering, stages, plan);
}

const char* RenderExecutor::PlanPanelGpuCompositeControlsCapture(
	const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	const RenderState& rendering, std::span<PreparedBindings* const> stages,
	PanelGpuScalarPlan& plan) const {
	return PlanPanelGpuCapture(PanelGpuScalarKind::CompositeControls, buffer, colors, depth, rendering, stages, plan);
}

const char* RenderExecutor::PlanPanelGpuCompositeUvCapture(
	const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	const RenderState& rendering, std::span<PreparedBindings* const> stages,
	PanelGpuScalarPlan& plan) const {
	return PlanPanelGpuCapture(PanelGpuScalarKind::CompositeUv, buffer, colors, depth, rendering, stages, plan);
}

const char* RenderExecutor::PlanPanelGpuCapture(
	PanelGpuScalarKind kind, const CommandBuffer& buffer, std::span<const RenderColorInfo> colors,
	ImageId depth, const RenderState& rendering, std::span<PreparedBindings* const> stages,
	PanelGpuScalarPlan& plan) const {
	plan = {};
	const auto* recipe = GetPanelGpuScalarRecipe(kind);
	if (recipe == nullptr) return "unsupported-recipe-kind";
	if (buffer.IsInvalid() || &buffer.GetContext() != &m_context) return "invalid-command-context";
	if (stages.empty() || stages.size() > 4u || colors.size() > RENDER_COLOR_ATTACHMENTS_MAX ||
	    rendering.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX)
		return "stage-or-attachment-count";
	if (rendering.depth_stencil_attachment.depth_clear || rendering.depth_stencil_attachment.stencil_clear ||
	    std::any_of(rendering.color_attachments.begin(), rendering.color_attachments.end(),
	                [](const auto& attachment) { return attachment.is_clear; }))
		return "rendering-load-clear";
	const PreparedBindings* pixel = nullptr;
	uint32_t pixel_index = 0;
	for (uint32_t i = 0; i < stages.size(); ++i) {
		const auto* prepared = stages[i];
		if (prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime)
			return "invalid-bound-stage";
		const auto& program = *prepared->runtime->program;
		if (program.stage == ShaderType::Pixel) {
			if (pixel != nullptr) return "multiple-pixel-stages";
			pixel = prepared;
			pixel_index = i;
		}
		if (program.has_address_writes ||
		    std::any_of(program.info.buffers.begin(), program.info.buffers.end(),
		                [](const auto& resource) { return resource.written || resource.atomic; }))
			return "bound-buffer-or-address-write";
		if (prepared->images.size() != program.info.images.size() ||
		    std::any_of(program.info.images.begin(), program.info.images.end(),
		                [](const auto& resource) { return resource.written || resource.atomic || resource.atomic64; }) ||
		    std::any_of(prepared->images.begin(), prepared->images.end(),
		                [](const auto& binding) { return binding.desc.type == TextureCache::BindingType::Storage; }))
			return "bound-storage-image-or-unknown-write";
	}
	if (pixel == nullptr || pixel->runtime->program->shader_hash != recipe->pixel_hash)
		return "missing-exact-pixel-recipe";
	const auto& program = *pixel->runtime->program;
	const auto& snapshot = *pixel->runtime->resources;
	if (program.info.buffers.size() <= recipe->resource || snapshot.buffers.size() <= recipe->resource)
		return "missing-recipe-resource";
	const auto& resource = program.info.buffers[recipe->resource];
	if (resource.descriptor_index != recipe->resource || resource.first_use_pc != recipe->first_pc || !resource.read ||
	    !resource.scalar || resource.written || resource.atomic || resource.formatted ||
	    resource.image_alias != ShaderRecompiler::IR::BufferResource::NoImageAlias ||
	    resource.indirect_root != ShaderRecompiler::IR::BufferResource::NoIndirectBuffer ||
	    !resource.indirect_resources.empty() || !program.info.user_data_registers.empty())
		return "recipe-mapping-or-access";
	if (pixel->buffer_sources.size() <= recipe->resource || pixel->buffers.size() <= recipe->resource)
		return "missing-native-binding";
	const auto& words = snapshot.buffers[recipe->resource];
	if (words.dword_count != 4u) return "descriptor-word-count";
	ShaderBufferResource descriptor {};
	std::memcpy(descriptor.fields, words.dwords.data(), sizeof(descriptor.fields));
	const auto& source = pixel->buffer_sources[recipe->resource];
	const auto& native = pixel->buffers[recipe->resource];
	const auto offset_word = uint64_t {program.bindings.memory_offset_dword};
	if (offset_word >= pixel->shader_data.size()) return "missing-packed-offset";
	const uint32_t packed = pixel->shader_data[static_cast<size_t>(offset_word)];
	if (((packed >> (8u * recipe->resource)) & 255u) != 0u) return "nonzero-packed-offset";
	if ((descriptor.fields[1] & 0xffff0000u) != 0x00100000u || descriptor.fields[3] != 0x0004dfacu ||
	    descriptor.GetSize() != recipe->declared_size || source.address != descriptor.Base48() ||
	    source.size != recipe->declared_size ||
	    source.address == 0 || !GuestRange {source.address, source.size}.Valid())
		return "descriptor-or-resolved-range";
	if (native.buffer == nullptr || native.range != recipe->declared_size || (native.offset & 3u) != 0u)
		return "native-range";
	bool stream = false;
	const auto* owner = ResolvePanelScalarNativeOwner(m_context.GetBufferCache(), native,
	                                                 source.address, source.size, stream);
	if (owner == nullptr) return "unproved-native-owner";
	if (!PanelScalarRangeFits(recipe->selected_offset, recipe->selected_size, native.range))
		return "selected-native-extent";
	const uint64_t selected = native.offset + recipe->selected_offset;
	if (!PanelScalarRangeFits(selected, recipe->selected_size, owner->Size())) return "selected-owner-extent";
	const auto check_info = [&](const ImageInfo& info) -> const char* {
		for (const auto range: {info.data, info.stencil, info.metadata.range}) {
			if (range.Empty()) continue;
			if (!range.Valid()) return "invalid-attachment-range";
			if (ImageRangeOverlaps(source.address, source.size, range.address, range.size))
				return "attachment-overlap";
		}
		return nullptr;
	};
	const auto check_attachment = [&](ImageId id) -> const char* {
		const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
		if (!id || image == nullptr || !image->registered) return "unresolved-attachment";
		return check_info(image->info);
	};
	for (uint32_t slot = 0; slot < rendering.color_attachments.size(); ++slot) {
		const auto view = rendering.color_attachments[slot].image_view;
		if (view == nullptr) continue;
		if (slot >= rendering.num_color_attachments) return "unresolved-attachment";
		const auto color = std::find_if(colors.begin(), colors.end(),
		    [slot](const auto& target) { return target.target_slot == slot; });
		if (color == colors.end()) return "unresolved-attachment";
		const auto* image = m_context.GetTextureCache().m_slot_images.try_get(color->image_id);
		if (image == nullptr || std::none_of(image->views.begin(), image->views.end(),
		    [view](const auto& candidate) { return candidate.view == view; })) return "unresolved-attachment";
	}
	for (const auto& color: colors) {
		if (const auto* reason = check_attachment(color.image_id)) return reason;
		if (const auto* reason = check_info(color.desc.info)) return reason;
	}
	if (depth) {
		if (const auto* reason = check_attachment(depth)) return reason;
	} else if (rendering.depth_stencil_attachment.image_view != nullptr) {
		return "unresolved-attachment";
	}
	PanelGpuScalarPlan result;
	result.kind = kind;
	result.native = native;
	result.guest_address = source.address;
	result.guest_size = source.size;
	result.selected_offset = selected;
	result.owner_size = owner->Size();
	std::copy(std::begin(descriptor.fields), std::end(descriptor.fields), result.sharp.begin());
	result.stage_index = pixel_index;
	result.packed_offset_word = packed;
	result.stream = stream;
	plan = result;
	return nullptr;
}

bool RenderExecutor::QueuePanelGpuScalarReadback(
	CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	Common::UniqueFunction<void, bool, uint32_t>&& completion) {
	if (!completion || plan.kind != PanelGpuScalarKind::CommonRgb) return false;
	return QueuePanelGpuReadback(PanelGpuScalarKind::CommonRgb, buffer, plan,
	    [completion = std::move(completion)](bool success, PanelGpuScalarWords words) mutable {
		    completion(success, words[0]);
	    });
}

bool RenderExecutor::QueuePanelGpuHdrUvReadback(
	CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	Common::UniqueFunction<void, bool, PanelGpuScalarWords>&& completion) {
	return QueuePanelGpuReadback(PanelGpuScalarKind::HdrUv, buffer, plan, std::move(completion));
}

bool RenderExecutor::QueuePanelGpuCompositeControlsReadback(
	CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	Common::UniqueFunction<void, bool, PanelGpuCompositeControlWords>&& completion) {
	if (!completion || plan.kind != PanelGpuScalarKind::CompositeControls) return false;
	return QueuePanelGpuReadback(PanelGpuScalarKind::CompositeControls, buffer, plan,
	    [completion = std::move(completion)](bool success, PanelGpuScalarWords words) mutable {
		    PanelGpuCompositeControlWords controls {};
		    std::copy_n(words.begin(), controls.size(), controls.begin());
		    completion(success, controls);
	    });
}

bool RenderExecutor::QueuePanelGpuCompositeUvReadback(
	CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	Common::UniqueFunction<void, bool, PanelGpuScalarWords>&& completion) {
	return QueuePanelGpuReadback(PanelGpuScalarKind::CompositeUv, buffer, plan, std::move(completion));
}

bool RenderExecutor::QueuePanelGpuReadback(
	PanelGpuScalarKind kind, CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	Common::UniqueFunction<void, bool, PanelGpuScalarWords>&& completion) {
	const auto* recipe = GetPanelGpuScalarRecipe(kind);
	if (recipe == nullptr || plan.kind != kind || recipe->selected_size > sizeof(PanelGpuScalarWords)) return false;
	if (buffer.IsInvalid() || &buffer.GetContext() != &m_context || !completion ||
	    plan.guest_size != recipe->declared_size || plan.native.range != recipe->declared_size ||
	    (plan.native.offset & 3u) != 0u ||
	    ((plan.packed_offset_word >> (8u * recipe->resource)) & 255u) != 0u)
		return false;
	ShaderBufferResource descriptor {};
	std::copy(plan.sharp.begin(), plan.sharp.end(), std::begin(descriptor.fields));
	if (descriptor.Base48() != plan.guest_address || descriptor.GetSize() != recipe->declared_size ||
	    (descriptor.fields[1] & 0xffff0000u) != 0x00100000u || descriptor.fields[3] != 0x0004dfacu)
		return false;
	bool stream = false;
	const auto* owner = ResolvePanelScalarNativeOwner(m_context.GetBufferCache(), plan.native,
	                                                 plan.guest_address, plan.guest_size, stream);
	if (owner == nullptr || owner->Size() != plan.owner_size || stream != plan.stream ||
	    !PanelScalarRangeFits(recipe->selected_offset, recipe->selected_size, plan.native.range) ||
	    plan.selected_offset != plan.native.offset + recipe->selected_offset ||
	    !PanelScalarRangeFits(plan.selected_offset, recipe->selected_size, owner->Size()))
		return false;
	// Validate everything before allocation or commands. A failed diagnostic allocation
	// must not terminate the emulator or consume a capture slot.
	auto readback = std::make_unique<PanelInputReadback>();
	readback->allocator = buffer.GetGraphics().allocator;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = recipe->selected_size;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	if (vmaCreateBuffer(readback->allocator, &buffer_info, &allocation_info,
	                    &readback->buffer, &readback->allocation, nullptr) != VK_SUCCESS)
		return false;
	buffer.EndRendering();
	const auto command = buffer.Handle();
	vk::BufferMemoryBarrier2 before {};
	before.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eHost;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eHostWrite;
	before.dstStageMask = vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eFragmentShader;
	before.dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eShaderStorageRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer = plan.native.buffer;
	before.offset = plan.selected_offset;
	before.size = recipe->selected_size;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers = &before;
	command.pipelineBarrier2(dependency);
	const vk::BufferCopy copy {plan.selected_offset, 0, recipe->selected_size};
	command.copyBuffer(plan.native.buffer, readback->buffer, 1, &copy);
	std::array<vk::BufferMemoryBarrier2, 2> after {};
	after[0] = before;
	after[0].srcStageMask = vk::PipelineStageFlagBits2::eCopy;
	after[0].srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	after[0].dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	after[0].dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	after[1].srcStageMask = vk::PipelineStageFlagBits2::eCopy;
	after[1].srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	after[1].dstStageMask = vk::PipelineStageFlagBits2::eHost;
	after[1].dstAccessMask = vk::AccessFlagBits2::eHostRead;
	after[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after[1].buffer = readback->buffer;
	after[1].offset = 0;
	after[1].size = recipe->selected_size;
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(after.size());
	dependency.pBufferMemoryBarriers = after.data();
	command.pipelineBarrier2(dependency);
	// Source allocations already live through this tick: registered owners retire
	// through DeferOperation, and stream Commit guards reservation reuse. Retain only
	// the dedicated destination and callback; never retain a cache or mapped pointer.
	m_context.GetCommandScheduler().DeferOperation(
	    [readback = std::move(readback), completion = std::move(completion), bytes = recipe->selected_size]() mutable {
		    PanelGpuScalarWords words {};
		    const bool success = vmaMapMemory(readback->allocator, readback->allocation,
		                                     &readback->mapped) == VK_SUCCESS &&
		        readback->mapped != nullptr &&
		        vmaInvalidateAllocation(readback->allocator, readback->allocation, 0, bytes) == VK_SUCCESS;
		    if (success) std::memcpy(words.data(), readback->mapped, bytes);
		    completion(success, words);
	    });
	return true;
}

uint32_t RenderExecutor::CapturePanelGpuScalarBeforeDraw(
	uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	const DrawEmitInfo& emit, const DrawRenderState& state, const RenderState& rendering,
	std::span<PreparedBindings* const> stages) {
	const auto& options = GetPanelGpuScalarCaptureOptions();
	if (options.composite)
		return CapturePanelGpuCompositeBeforeDraw(submit_id, buffer, draw, emit, state, rendering, stages);
	const uint64_t max_raw_bytes = options.hdr_uv ? 104u : 8u;
	if (!options.enabled || m_gpu_scalar_capture.index >= options.milestones.size() ||
	    !PanelScalarRangeFits(m_gpu_scalar_capture.bytes, 4u, max_raw_bytes) || m_gpu_scalar_decisions >= 64u ||
	    !GuestGpu::IsGpuThread() || m_upstream_source_count == 0)
		return 0;
	const auto frame = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
	const auto milestone = options.milestones[m_gpu_scalar_capture.index];
	if (frame > 4096u || frame < milestone || frame == m_gpu_scalar_last_frame) return 0;
	const bool exact_pixel = std::any_of(stages.begin(), stages.end(), [](const auto* prepared) {
		return prepared != nullptr && prepared->runtime != nullptr && *prepared->runtime &&
		       prepared->runtime->program->stage == ShaderType::Pixel &&
		       prepared->runtime->program->shader_hash == 0x83b1c2d863c50cf3ull;
	});
	if (!exact_pixel) return 0;
	const auto colors = std::span {state.color_info, state.color_count};
	const auto match = MatchPanelProducerWrites(colors, stages);
	if ((match.images & PanelUpstreamWatchMask) == 0) return 0;
	++m_gpu_scalar_decisions;
	PanelGpuScalarPlan plan;
	const auto* reason = PlanPanelGpuScalarCapture(buffer, colors, state.depth_info.image_id,
	                                              rendering, stages, plan);
	if (reason != nullptr) {
		LOGF("PanelProducer GPU scalar decision: frame=%u submit=%" PRIu64
		     " milestone=%u decision=%u state=skipped reason=%s hdr_uv_state=not-attempted\n",
		     frame, submit_id, milestone, m_gpu_scalar_decisions, reason);
		return 0;
	}
	const uint32_t sequence = m_gpu_scalar_capture.index + 1u;
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	const auto path = fmt::format("{}-scalar{}-frame{}-submit{}-tick{}.bin",
	                             options.prefix, sequence, frame, submit_id, tick);
	std::string metadata = fmt::format(
	    "PanelGpuScalar sequence={} milestone={} frame={} submit={} tick={} timing=before-draw "
	    "origin=gpu-buffer-copy kind=common-rgb role=pre-LUT-common-RGB-multiplier bytes=4 "
	    "stage={} ps=83b1c2d863c50cf3 resource=1 binding=1 first_pc=40 source_images={:x}\n"
	    "draw={} count={} instances={} vertex_offset={} first_vertex={} first_instance={}\n"
	    "sharp={:08x},{:08x},{:08x},{:08x} packed_offset_word={:08x} "
	    "guest={:016x} resolved_bytes={} native={:016x} offset={} range={} selected_byte_offset=2168 "
	    "selected_native_offset={} owner_size={} owner={}\n",
	    sequence, milestone, frame, submit_id, tick, plan.stage_index, match.images,
	    draw.Name(), draw.index_count, draw.instance_count, emit.vertex_offset, emit.first_vertex, emit.first_instance,
	    plan.sharp[0], plan.sharp[1], plan.sharp[2], plan.sharp[3], plan.packed_offset_word,
	    plan.guest_address, plan.guest_size, PanelNativeHandle(plan.native.buffer), plan.native.offset, plan.native.range,
	    plan.selected_offset, plan.owner_size, plan.stream ? "stream" : "registered-cache");
	const bool queued = QueuePanelGpuScalarReadback(buffer, plan,
	    [path, metadata = std::move(metadata), sequence, frame, tick](bool readback_ok, uint32_t bits) {
		    uint32_t written = 0;
		    bool saved = false;
		    if (readback_ok) {
			    Common::File raw;
			    if (raw.Create(path)) {
				    raw.Write(&bits, sizeof(bits), &written);
				    saved = written == sizeof(bits) && raw.Flush();
			    }
		    }
		    auto text = metadata + fmt::format("gpu_completed=1 readback_ok={} saved={} written={}",
		                                       readback_ok, saved, written);
		    if (readback_ok) text += fmt::format(" dword={:08x} float={}", bits, std::bit_cast<float>(bits));
		    text += "\n";
		    Common::File info;
		    bool metadata_saved = false;
		    if (info.Create(path + ".txt")) {
			    uint32_t count = 0;
			    info.Write(text.data(), static_cast<uint32_t>(text.size()), &count);
			    metadata_saved = count == text.size() && info.Flush();
		    }
		    LOGF("PanelProducer GPU scalar complete: sequence=%u frame=%u tick=%" PRIu64
		         " file=%s gpu_completed=1 readback_ok=%d saved=%d metadata=%d bytes=%u\n",
		         sequence, frame, tick, path.c_str(), readback_ok, saved, metadata_saved, written);
	    });
	// The common scalar owns the set. The companion can only be queued after the
	// primary copy succeeded, and shares its frame, tick, sequence and decision row.
	bool hdr_uv_queued = false;
	const char* hdr_uv_reason = options.hdr_uv ? "primary-not-queued" : "disabled";
	PanelGpuScalarPlan hdr_uv_plan;
	if (queued && options.hdr_uv) {
		hdr_uv_reason = PanelScalarRangeFits(m_gpu_scalar_capture.bytes + 4u, 48u, max_raw_bytes)
		                    ? PlanPanelGpuHdrUvCapture(buffer, colors, state.depth_info.image_id,
		                                              rendering, stages, hdr_uv_plan)
		                    : "raw-byte-budget";
		if (hdr_uv_reason == nullptr) {
			const auto hdr_uv_path = fmt::format("{}-scalar{}-hdr-uv-frame{}-submit{}-tick{}.bin",
			                                     options.prefix, sequence, frame, submit_id, tick);
			std::string hdr_uv_metadata = fmt::format(
			    "PanelGpuScalar sequence={} milestone={} frame={} submit={} tick={} timing=before-draw "
			    "origin=gpu-buffer-copy kind=hdr-uv role=HDR-input1-UV-transform bytes=48 primary_queued=1 "
			    "stage={} ps=83b1c2d863c50cf3 resource=0 binding=0 first_pc=10 source_images={:x}\n"
			    "draw={} count={} instances={} vertex_offset={} first_vertex={} first_instance={}\n"
			    "sharp={:08x},{:08x},{:08x},{:08x} packed_offset_word={:08x} "
			    "guest={:016x} resolved_bytes={} native={:016x} offset={} range={} selected_byte_offset=0 "
			    "selected_native_offset={} owner_size={} owner={}\n"
			    "interpretation=raw-dwords-authoritative float_words=0,1,2,3,10,11 uint_words=8,9 "
			    "remaining_words=uninterpreted\n",
			    sequence, milestone, frame, submit_id, tick, hdr_uv_plan.stage_index, match.images,
			    draw.Name(), draw.index_count, draw.instance_count, emit.vertex_offset, emit.first_vertex, emit.first_instance,
			    hdr_uv_plan.sharp[0], hdr_uv_plan.sharp[1], hdr_uv_plan.sharp[2], hdr_uv_plan.sharp[3],
			    hdr_uv_plan.packed_offset_word, hdr_uv_plan.guest_address, hdr_uv_plan.guest_size,
			    PanelNativeHandle(hdr_uv_plan.native.buffer), hdr_uv_plan.native.offset, hdr_uv_plan.native.range,
			    hdr_uv_plan.selected_offset, hdr_uv_plan.owner_size, hdr_uv_plan.stream ? "stream" : "registered-cache");
			hdr_uv_queued = QueuePanelGpuHdrUvReadback(buffer, hdr_uv_plan,
			    [hdr_uv_path, metadata = std::move(hdr_uv_metadata), sequence, frame, tick]
			    (bool readback_ok, PanelGpuScalarWords words) {
				    uint32_t written = 0;
				    bool saved = false;
				    if (readback_ok) {
					    Common::File raw;
					    if (raw.Create(hdr_uv_path)) {
						    raw.Write(words.data(), 48u, &written);
						    saved = written == 48u && raw.Flush();
					    }
				    }
				    auto text = metadata + fmt::format("gpu_completed=1 readback_ok={} saved={} written={}",
				                                       readback_ok, saved, written);
				    if (readback_ok) {
					    text += " dwords=";
					    for (uint32_t i = 0; i < words.size(); ++i)
						    text += fmt::format("{}{:08x}", i == 0 ? "" : ",", words[i]);
					    text += fmt::format(" float_words_0_1_2_3={},{},{},{} uint_words_8_9={},{} "
					                        "float_words_10_11={},{}",
					                        std::bit_cast<float>(words[0]), std::bit_cast<float>(words[1]),
					                        std::bit_cast<float>(words[2]), std::bit_cast<float>(words[3]),
					                        words[8], words[9], std::bit_cast<float>(words[10]),
					                        std::bit_cast<float>(words[11]));
				    }
				    text += "\n";
				    Common::File info;
				    bool metadata_saved = false;
				    if (info.Create(hdr_uv_path + ".txt")) {
					    uint32_t count = 0;
					    info.Write(text.data(), static_cast<uint32_t>(text.size()), &count);
					    metadata_saved = count == text.size() && info.Flush();
				    }
				    LOGF("PanelProducer GPU scalar complete: sequence=%u kind=hdr-uv frame=%u tick=%" PRIu64
				         " file=%s gpu_completed=1 readback_ok=%d saved=%d metadata=%d bytes=%u\n",
				         sequence, frame, tick, hdr_uv_path.c_str(), readback_ok, saved, metadata_saved, written);
			    });
			hdr_uv_reason = hdr_uv_queued ? "ok" : "owner-changed-or-allocation-failed";
		}
	}
	LOGF("PanelProducer GPU scalar decision: frame=%u submit=%" PRIu64
	     " milestone=%u decision=%u sequence=%u state=%s reason=%s native=%016" PRIx64
	     " selected_offset=%" PRIu64 " bytes=4 tick=%" PRIu64
	     " hdr_uv_state=%s hdr_uv_reason=%s hdr_uv_native=%016" PRIx64
	     " hdr_uv_selected_offset=%" PRIu64 " hdr_uv_bytes=%u\n",
	     frame, submit_id, milestone, m_gpu_scalar_decisions, sequence, queued ? "queued" : "skipped",
	     queued ? "ok" : "owner-changed-or-allocation-failed", PanelNativeHandle(plan.native.buffer),
	     plan.selected_offset, tick, hdr_uv_queued ? "queued" : "skipped", hdr_uv_reason,
	     PanelNativeHandle(hdr_uv_plan.native.buffer), hdr_uv_plan.selected_offset, hdr_uv_queued ? 48u : 0u);
	if (!queued) return 0;
	++m_gpu_scalar_capture.index;
	m_gpu_scalar_capture.bytes += 4u + (hdr_uv_queued ? 48u : 0u);
	m_gpu_scalar_last_frame = frame;
	return sequence;
}

uint32_t RenderExecutor::CapturePanelGpuCompositeBeforeDraw(
	uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	const DrawEmitInfo& emit, const DrawRenderState& state, const RenderState& rendering,
	std::span<PreparedBindings* const> stages) {
	const auto& options = GetPanelGpuScalarCaptureOptions();
	constexpr uint64_t max_raw_bytes = 152u;
	if (!options.enabled || !options.composite || m_gpu_scalar_capture.index >= options.milestones.size() ||
	    !PanelScalarRangeFits(m_gpu_scalar_capture.bytes, 28u, max_raw_bytes) || m_gpu_scalar_decisions >= 64u ||
	    !GuestGpu::IsGpuThread() || m_upstream_source_count == 0)
		return 0;
	const auto frame = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
	const auto milestone = options.milestones[m_gpu_scalar_capture.index];
	if (frame > 4096u || frame < milestone || frame == m_gpu_scalar_last_frame) return 0;
	const bool exact_pixel = std::any_of(stages.begin(), stages.end(), [](const auto* prepared) {
		return prepared != nullptr && prepared->runtime != nullptr && *prepared->runtime &&
		       prepared->runtime->program->stage == ShaderType::Pixel &&
		       prepared->runtime->program->shader_hash == 0x8f4f493f4d30416dull;
	});
	if (!exact_pixel) return 0;
	const auto colors = std::span {state.color_info, state.color_count};
	const auto match = MatchPanelProducerWrites(colors, stages);
	if ((match.images & PanelUpstreamWatchMask) == 0) return 0;
	++m_gpu_scalar_decisions;
	PanelGpuScalarPlan plan;
	const auto* reason = PlanPanelGpuCompositeControlsCapture(buffer, colors, state.depth_info.image_id,
	                                                        rendering, stages, plan);
	if (reason != nullptr) {
		LOGF("PanelProducer GPU scalar decision: mode=composite frame=%u submit=%" PRIu64
		     " milestone=%u decision=%u state=skipped reason=%s composite_uv_state=not-attempted\n",
		     frame, submit_id, milestone, m_gpu_scalar_decisions, reason);
		return 0;
	}
	const uint32_t sequence = m_gpu_scalar_capture.index + 1u;
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	const auto queue_capture = [&](const PanelGpuScalarPlan& selected, bool uv) {
		const auto kind = uv ? PanelGpuScalarKind::CompositeUv : PanelGpuScalarKind::CompositeControls;
		if (selected.kind != kind) return false;
		const uint32_t bytes = uv ? 48u : 28u;
		const char* kind_name = uv ? "composite-uv" : "composite-controls";
		const char* role = uv ? "scene-input13-UV-constants" : "scene-mix-distortion-bias-controls";
		const auto path = fmt::format("{}-scalar{}-{}-frame{}-submit{}-tick{}.bin",
		                             options.prefix, sequence, kind_name, frame, submit_id, tick);
		std::string metadata = fmt::format(
		    "PanelGpuScalar sequence={} milestone={} frame={} submit={} tick={} timing=before-draw "
		    "origin=gpu-buffer-copy kind={} role={} bytes={}",
		    sequence, milestone, frame, submit_id, tick, kind_name, role, bytes);
		if (uv) metadata += " primary_queued=1";
		metadata += fmt::format(
		    " stage={} ps=8f4f493f4d30416d resource={} binding={} first_pc={:x} source_images={:x}\n"
		    "draw={} count={} instances={} vertex_offset={} first_vertex={} first_instance={}\n"
		    "sharp={:08x},{:08x},{:08x},{:08x} packed_offset_word={:08x} "
		    "guest={:016x} resolved_bytes={} native={:016x} offset={} range={} selected_byte_offset=0 "
		    "selected_native_offset={} owner_size={} owner={}\n",
		    selected.stage_index, uv ? 0u : 3u, uv ? 0u : 3u, uv ? 0x10u : 0x1d0u, match.images,
		    draw.Name(), draw.index_count, draw.instance_count, emit.vertex_offset, emit.first_vertex, emit.first_instance,
		    selected.sharp[0], selected.sharp[1], selected.sharp[2], selected.sharp[3], selected.packed_offset_word,
		    selected.guest_address, selected.guest_size, PanelNativeHandle(selected.native.buffer),
		    selected.native.offset, selected.native.range, selected.selected_offset, selected.owner_size,
		    selected.stream ? "stream" : "registered-cache");
		metadata += uv
		    ? "interpretation=raw-dwords-authoritative float_words=0,1,2,3,4,5,6,7,10,11 uint_words=8,9 "
		      "remap-branches-and-distortion-unmeasured=1 sampled-RGB-unproved=1\n"
		    : "interpretation=raw-dwords-authoritative float_words=0,4,6 remaining_words=uninterpreted "
		      "t_word=0 distortion_mix_word=4 bias_word=6\n";
		auto completion = [path, metadata = std::move(metadata), sequence, frame, tick, bytes, kind_name]
		                  (bool readback_ok, auto words) {
			uint32_t written = 0;
			bool saved = false;
			if (readback_ok && sizeof(words) == bytes) {
				Common::File raw;
				if (raw.Create(path)) {
					raw.Write(words.data(), bytes, &written);
					saved = written == bytes && raw.Flush();
				}
			}
			auto text = metadata + fmt::format("gpu_completed=1 readback_ok={} saved={} written={}",
			                                   readback_ok, saved, written);
			if (readback_ok) {
				text += " dwords=";
				for (uint32_t i = 0; i < words.size(); ++i)
					text += fmt::format("{}{:08x}", i == 0 ? "" : ",", words[i]);
			}
			text += "\n";
			Common::File info;
			bool metadata_saved = false;
			if (info.Create(path + ".txt")) {
				uint32_t count = 0;
				info.Write(text.data(), static_cast<uint32_t>(text.size()), &count);
				metadata_saved = count == text.size() && info.Flush();
			}
			LOGF("PanelProducer GPU scalar complete: sequence=%u kind=%s frame=%u tick=%" PRIu64
			     " file=%s gpu_completed=1 readback_ok=%d saved=%d metadata=%d bytes=%u\n",
			     sequence, kind_name, frame, tick, path.c_str(), readback_ok, saved, metadata_saved, written);
		};
		if (uv) return QueuePanelGpuCompositeUvReadback(buffer, selected, std::move(completion));
		return QueuePanelGpuCompositeControlsReadback(buffer, selected, std::move(completion));
	};
	const bool queued = queue_capture(plan, false);
	bool uv_queued = false;
	const char* uv_reason = "primary-not-queued";
	PanelGpuScalarPlan uv_plan;
	if (queued) {
		uv_reason = PanelScalarRangeFits(m_gpu_scalar_capture.bytes + 28u, 48u, max_raw_bytes)
		                ? PlanPanelGpuCompositeUvCapture(buffer, colors, state.depth_info.image_id,
		                                                rendering, stages, uv_plan)
		                : "raw-byte-budget";
		if (uv_reason == nullptr) {
			uv_queued = queue_capture(uv_plan, true);
			uv_reason = uv_queued ? "ok" : "owner-changed-or-allocation-failed";
		}
	}
	LOGF("PanelProducer GPU scalar decision: mode=composite frame=%u submit=%" PRIu64
	     " milestone=%u decision=%u sequence=%u state=%s reason=%s native=%016" PRIx64
	     " selected_offset=%" PRIu64 " bytes=28 tick=%" PRIu64
	     " composite_uv_state=%s composite_uv_reason=%s composite_uv_native=%016" PRIx64
	     " composite_uv_selected_offset=%" PRIu64 " composite_uv_bytes=%u\n",
	     frame, submit_id, milestone, m_gpu_scalar_decisions, sequence, queued ? "queued" : "skipped",
	     queued ? "ok" : "owner-changed-or-allocation-failed", PanelNativeHandle(plan.native.buffer),
	     plan.selected_offset, tick, uv_queued ? "queued" : "skipped", uv_reason,
	     PanelNativeHandle(uv_plan.native.buffer), uv_plan.selected_offset, uv_queued ? 48u : 0u);
	if (!queued) return 0;
	++m_gpu_scalar_capture.index;
	m_gpu_scalar_capture.bytes += 28u + (uv_queued ? 48u : 0u);
	m_gpu_scalar_last_frame = frame;
	return sequence;
}

bool RenderExecutor::PanelProducerTraceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_VR_PANEL_PRODUCER_TRACE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

void RenderExecutor::ObservePanelSources(const CommandBuffer& buffer,
	                                      std::span<PreparedBindings* const> stages) {
	if (!PanelProducerTraceEnabled()) return;
	const auto frame = static_cast<uint32_t>(buffer.GetContext().GetGpu().GetFrameNum());
	if (frame > 4096u) return;
	auto& cache = m_context.GetTextureCache();
	for (const auto* stage: stages) {
		const auto& program = *stage->runtime->program;
		const auto& snapshot = *stage->runtime->resources;
		const auto count = std::min({stage->images.size(), program.info.images.size(), snapshot.images.size()});
		for (uint32_t i = 0; i < count; ++i) {
			const auto& resource = program.info.images[i];
			const auto& binding = stage->images[i];
			if (!resource.read || resource.resource_class != ShaderRecompiler::IR::ImageResourceClass::Sampled)
				continue;
			const auto* image = cache.m_slot_images.try_get(binding.image_id);
			if (image == nullptr || image->backing.image == nullptr) continue;
			const auto found = std::find_if(image->views.begin(), image->views.end(),
			    [&](const auto& view) { return view.view == binding.image_view; });
			if (found == image->views.end()) continue;
			const auto& view = found->info;
			uint32_t slot = m_panel_source_count;
			bool unchanged = false;
			for (uint32_t n = 0; n < m_panel_source_count; ++n) {
				const auto& old = m_panel_sources[n];
				if (old.native == image->backing.image && old.image_id == binding.image_id &&
				    old.view.aspect == view.aspect && old.view.base_level == view.base_level &&
				    old.view.level_count == view.level_count && old.view.base_layer == view.base_layer &&
				    old.view.layer_count == view.layer_count) {
					slot = n;
					unchanged = old.data == image->info.data && old.metadata == image->info.metadata.range &&
					            old.stencil == image->info.stencil;
					break;
				}
				const auto* previous = cache.m_slot_images.try_get(old.image_id);
				if (previous == nullptr || previous->backing.image != old.native ||
				    (old.data.address == image->info.data.address && old.native != image->backing.image)) slot = n;
			}
			if (unchanged) continue;
			if (slot >= PanelPrimaryWatchCapacity) {
				if (m_panel_source_logs++ < 16u) LOGF("PanelProducer watch limit: cap=8 new_guest=0x%016" PRIx64 "\n", image->info.data.address);
				continue;
			}
			if (m_producer_first_frame == UINT32_MAX) {
				m_producer_first_frame = frame;
				LOGF("PanelProducer enabled: learned-from=actual-panel-sampled-bindings frame_limit=4096 "
				     "events_per_frame=64 inputs_per_event=16 watch_cap=8 "
				     "uncovered=pre-discovery-writes,cache-color-materialization,cpu-uploads,depth-transfer-shortcuts,unattributed-address-writes,non-front-dynamic-mip-views "
				     "commands-do-not-prove-pixel-coverage\n");
			}
			m_panel_sources[slot] = {binding.image_id, image->backing.image, view, image->info.data,
			                        image->info.stencil, image->info.metadata.range};
			if (slot == m_panel_source_count) ++m_panel_source_count;
			if (m_panel_source_logs++ < 16u) {
				const auto& words = snapshot.images[i].dwords;
				LOGF("PanelProducer watch: slot=%u frame=%u id=%u native=0x%016" PRIx64
				     " guest=0x%016" PRIx64 " bytes=%" PRIu64 " metadata=0x%016" PRIx64
				     " meta_bytes=%" PRIu64 " aspect=0x%x mip=%u:%u layer=%u:%u dynamic_views=%zu "
				     "tsharp=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
				     slot, frame, binding.image_id.index, PanelNativeHandle(image->backing.image),
				     image->info.data.address, image->info.data.size, image->info.metadata.range.address,
				     image->info.metadata.range.size, static_cast<uint32_t>(view.aspect), view.base_level,
				     view.level_count, view.base_layer, view.layer_count, binding.mip_views.size(),
				     words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
			}
		}
	}
}

void RenderExecutor::LogPanelUpstreamSkip(uint32_t frame, uint32_t reason_bit, const char* reason) {
	++m_upstream_skipped;
	if ((m_upstream_skip_reasons & reason_bit) != 0) return;
	m_upstream_skip_reasons |= reason_bit;
	LOGF("PanelProducer upstream skipped: frame=%u reason=%s skipped=%u reason_mask=0x%x\n",
	     frame, reason, m_upstream_skipped, m_upstream_skip_reasons);
}

void RenderExecutor::ObservePanelUpstreamSources(const CommandBuffer& buffer,
	    std::span<PreparedBindings* const> stages, uint32_t parent_mask) {
	const auto& options = GetPanelUpstreamOptions();
	parent_mask &= PanelPrimaryWatchMask;
	if (!options.enabled || parent_mask == 0) return;
	const auto frame = static_cast<uint32_t>(buffer.GetContext().GetGpu().GetFrameNum());
	const PreparedBindings* pixel = nullptr;
	for (const auto* stage: stages) {
		if (stage->runtime->program->stage == ShaderType::Pixel &&
		    stage->runtime->program->shader_hash == options.parent_ps_hash) pixel = stage;
	}
	if (pixel == nullptr) {
		LogPanelUpstreamSkip(frame, 1u << 0u, "parent-pixel-hash-filter");
		return;
	}
	auto& cache = m_context.GetTextureCache();
	const auto& program = *pixel->runtime->program;
	const auto& snapshot = *pixel->runtime->resources;
	const auto count = std::min({pixel->images.size(), program.info.images.size(), snapshot.images.size()});
	uint32_t sampled = 0;
	for (uint32_t i = 0; i < count; ++i) {
		const auto& resource = program.info.images[i];
		if (!resource.read || resource.written || resource.resource_class != ShaderRecompiler::IR::ImageResourceClass::Sampled)
			continue;
		++sampled;
		const auto& binding = pixel->images[i];
		const auto* image = cache.m_slot_images.try_get(binding.image_id);
		if (image == nullptr || !image->registered || image->backing.image == nullptr || image->info.data.Empty()) {
			LogPanelUpstreamSkip(frame, 1u << 1u, "missing-or-unregistered-native-source");
			continue;
		}
		if (binding.mip_views.size() > 1u) {
			LogPanelUpstreamSkip(frame, 1u << 2u, "dynamic-mip-views-not-promoted");
			continue;
		}
		const auto found = std::find_if(image->views.begin(), image->views.end(),
		    [&](const auto& native) { return native.view == binding.image_view; });
		if (found == image->views.end()) {
			LogPanelUpstreamSkip(frame, 1u << 3u, "actual-bound-view-not-found");
			continue;
		}
		const auto& view = found->info;
		if ((MatchPanelSourceImage(binding.image_id, view) & PanelPrimaryWatchMask) != 0) {
			LogPanelUpstreamSkip(frame, 1u << 4u, "source-already-primary-no-cycle");
			continue;
		}
		uint32_t slot = PanelPrimaryWatchCapacity + m_upstream_source_count;
		uint32_t stale = UINT32_MAX;
		uint32_t oldest = PanelPrimaryWatchCapacity;
		bool same_identity = false;
		for (uint32_t n = PanelPrimaryWatchCapacity; n < PanelSourceLimit(); ++n) {
			const auto& old = m_panel_sources[n];
			if (old.native == image->backing.image && old.image_id == binding.image_id &&
			    old.view.aspect == view.aspect && old.view.base_level == view.base_level &&
			    old.view.level_count == view.level_count && old.view.base_layer == view.base_layer &&
			    old.view.layer_count == view.layer_count) {
				slot = n;
				same_identity = true;
				break;
			}
			const auto* current = cache.m_slot_images.try_get(old.image_id);
			if (current == nullptr || !current->registered || current->backing.image != old.native || current->info.data != old.data)
				stale = n;
			if (old.last_observed_frame < m_panel_sources[oldest].last_observed_frame) oldest = n;
		}
		const char* action = same_identity ? "refresh-association" : "learn";
		if (!same_identity && stale != UINT32_MAX) {
			slot = stale;
			action = "replace-stale";
		} else if (!same_identity && slot >= m_panel_sources.size()) {
			slot = oldest;
			action = "replace-oldest-full-pool";
			LogPanelUpstreamSkip(frame, 1u << 5u, "child-pool-full-replacing-exact-observed-binding");
		}
		auto& watch = m_panel_sources[slot];
		const uint32_t parents = same_identity ? watch.parent_mask | parent_mask : parent_mask;
		if (same_identity && watch.data == image->info.data && watch.stencil == image->info.stencil &&
		    watch.metadata == image->info.metadata.range && watch.view == view && watch.parent_mask == parents) {
			watch.last_observed_frame = frame;
			continue;
		}
		const auto replaced_native = watch.native;
		const auto replaced_id = watch.image_id;
		watch = {binding.image_id, image->backing.image, view, image->info.data, image->info.stencil,
		         image->info.metadata.range, parents, frame};
		if (slot == PanelPrimaryWatchCapacity + m_upstream_source_count) ++m_upstream_source_count;
		if (!same_identity) {
			if (m_upstream_discovery_count < m_upstream_discovery_frames.size()) {
				m_upstream_discovery_frames[m_upstream_discovery_count++] = frame;
			} else {
				LogPanelUpstreamSkip(frame, 1u << 6u, "discovery-window-cap-fixed-milestones-remain");
			}
		}
		if (m_upstream_watch_logs++ < 32u) {
			const auto& words = snapshot.images[i].dwords;
			LOGF("PanelProducer watch: slot=%u frame=%u ancestry=1 parent_mask=0x%x parent_mask_kind=historical-primary-slots "
			     "observed_parent_mask=0x%x parent_ps=%016" PRIx64
			     " action=%s old_id=%u old_native=0x%016" PRIx64 " id=%u native=0x%016" PRIx64
			     " sampled_view=0x%016" PRIx64 " guest=0x%016" PRIx64 " bytes=%" PRIu64
			     " metadata=0x%016" PRIx64 " meta_bytes=%" PRIu64 " stencil=0x%016" PRIx64 " stencil_bytes=%" PRIu64
			     " aspect=0x%x mip=%u:%u layer=%u:%u view_fmt=%u image=%u binding=%u "
			     "tsharp=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
			     slot, frame, parents, parent_mask, options.parent_ps_hash, action, replaced_id.index, PanelNativeHandle(replaced_native),
			     binding.image_id.index, PanelNativeHandle(image->backing.image), PanelNativeHandle(binding.image_view),
			     image->info.data.address, image->info.data.size, image->info.metadata.range.address, image->info.metadata.range.size,
			     image->info.stencil.address, image->info.stencil.size, static_cast<uint32_t>(view.aspect), view.base_level,
			     view.level_count, view.base_layer, view.layer_count, static_cast<uint32_t>(view.format), i, resource.descriptor_index,
			     words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
		} else if (m_upstream_watch_logs == 33u) {
			LOGF("PanelProducer upstream watch log limit: cap=32 later-slot-identities-unreported\n");
		}
	}
	if (sampled == 0) LogPanelUpstreamSkip(frame, 1u << 7u, "filtered-parent-has-no-readonly-sampled-input");
}

uint32_t RenderExecutor::MatchPanelSourceImage(ImageId id, const ImageViewInfo& view) const {
	auto& cache = m_context.GetTextureCache();
	const auto* image = cache.m_slot_images.try_get(id);
	// New ancestry evidence cannot use an image already unregistered for deferred deletion.
	// Keep the original direct-bound-image diagnostic behavior when this option is unset.
	const bool require_registered = GetPanelUpstreamOptions().enabled;
	if (image == nullptr || (require_registered && !image->registered)) return 0;
	uint32_t matches = 0;
	for (uint32_t i = 0; i < PanelSourceLimit(); ++i) {
		const auto& watch = m_panel_sources[i];
		if (watch.native == nullptr) continue;
		const auto* current = cache.m_slot_images.try_get(watch.image_id);
		if (current == nullptr || (require_registered && !current->registered) ||
		    current->backing.image != watch.native || current->info.data != watch.data ||
		    image->backing.image != watch.native || !(view.aspect & watch.view.aspect)) continue;
		if (uint64_t {view.base_level} < uint64_t {watch.view.base_level} + watch.view.level_count &&
		    uint64_t {watch.view.base_level} < uint64_t {view.base_level} + view.level_count &&
		    uint64_t {view.base_layer} < uint64_t {watch.view.base_layer} + watch.view.layer_count &&
		    uint64_t {watch.view.base_layer} < uint64_t {view.base_layer} + view.layer_count) matches |= 1u << i;
	}
	return matches;
}

uint32_t RenderExecutor::MatchPanelSourceRange(GuestRange range, bool metadata) const {
	if (range.Empty()) return 0;
	auto& cache = m_context.GetTextureCache();
	uint32_t matches = 0;
	for (uint32_t i = 0; i < PanelSourceLimit(); ++i) {
		const auto& watch = m_panel_sources[i];
		if (watch.native == nullptr) continue;
		const auto* current = cache.m_slot_images.try_get(watch.image_id);
		if (current == nullptr || !current->registered || current->backing.image != watch.native ||
		    current->info.data != watch.data) continue;
		const std::array watched_ranges {metadata ? watch.metadata : watch.data, metadata ? GuestRange {} : watch.stencil};
		const std::array current_ranges {metadata ? current->info.metadata.range : current->info.data,
		                                 metadata ? GuestRange {} : current->info.stencil};
		for (size_t r = 0; r < watched_ranges.size(); ++r) {
			const auto target = watched_ranges[r];
			if (target == current_ranges[r] && !target.Empty() &&
			    ImageRangeOverlaps(range.address, range.size, target.address, target.size))
				matches |= 1u << i;
		}
	}
	return matches;
}

RenderExecutor::PanelProducerMatch RenderExecutor::MatchPanelProducerWrites(
	    std::span<const RenderColorInfo> colors, std::span<PreparedBindings* const> stages) const {
	PanelProducerMatch match {};
	auto& cache = m_context.GetTextureCache();
	for (const auto& color: colors) match.images |= MatchPanelSourceImage(color.image_id, color.desc.view_info);
	for (const auto* stage: stages) {
		const auto& program = *stage->runtime->program;
		match.unattributed_writes |= program.has_address_writes;
		for (uint32_t i = 0; i < std::min(stage->images.size(), program.info.images.size()); ++i) {
			if (!program.info.images[i].written) continue;
			const auto& binding = stage->images[i];
			const auto* image = cache.m_slot_images.try_get(binding.image_id);
			if (image == nullptr) continue;
			const auto found = std::find_if(image->views.begin(), image->views.end(),
			    [&](const auto& native) { return native.view == binding.image_view; });
			if (found != image->views.end()) match.images |= MatchPanelSourceImage(binding.image_id, found->info);
		}
		for (const auto& resource: program.info.buffers) {
			if (!resource.written) continue;
			if (resource.descriptor_index >= stage->buffer_sources.size()) {
				match.unattributed_writes |= resource.image_alias == ShaderRecompiler::IR::BufferResource::NoImageAlias;
				continue;
			}
			const auto& source = stage->buffer_sources[resource.descriptor_index];
			match.buffer_aliases |= MatchPanelSourceRange({source.address, source.size}, false);
			match.metadata_aliases |= MatchPanelSourceRange({source.address, source.size}, true);
		}
	}
	return match;
}

bool RenderExecutor::TracePanelProducerWindow(const CommandBuffer& buffer, uint32_t frame, const PanelProducerMatch& match) {
	const auto& upstream = GetPanelUpstreamOptions();
	if (frame != m_producer_frame) {
		if (m_producer_window) LOGF("PanelProducer window end: frame=%u matched=%u emitted=%u truncated=%u unattributed=%u\n",
		    m_producer_frame, m_producer_matches, m_producer_emitted, m_producer_matches - m_producer_emitted, m_producer_unattributed);
		if (m_producer_window && upstream.enabled) {
			LOGF("PanelProducer upstream window end: frame=%u native_writers=%u child_watches=%u "
			     "capture_sets=%u capture_bytes=%" PRIu64 " skipped=%u reason_mask=0x%x\n",
			     m_producer_frame, m_producer_upstream_native_matches, m_upstream_source_count,
			     m_producer_capture.index, m_producer_capture.bytes, m_upstream_skipped, m_upstream_skip_reasons);
		}
		m_producer_frame = frame;
		m_producer_matches = m_producer_emitted = m_producer_unattributed = 0;
		m_producer_upstream_native_matches = 0;
		m_producer_window = frame > m_producer_first_frame && frame - m_producer_first_frame <= 2u;
		for (const uint32_t selected: {128u, 256u, 512u, 1024u, 1536u, 2048u}) m_producer_window |= frame == selected;
		if (upstream.enabled) {
			for (const auto selected: upstream.capture_frames) m_producer_window |= frame == selected;
			// Discovery happens after this frame's input may already have been written.
			// Arm bounded complete later frames; do not imply earlier writers were observed.
			for (uint32_t i = 0; i < m_upstream_discovery_count; ++i) {
				const auto discovered = m_upstream_discovery_frames[i];
				m_producer_window |= frame > discovered && frame - discovered <= 3u;
			}
		}
		if (m_producer_window) {
			LOGF("PanelProducer window begin: frame=%u watches=%u cap=64\n",
			     frame, m_panel_source_count + m_upstream_source_count);
			if (upstream.enabled) {
				LOGF("PanelProducer upstream window begin: frame=%u primary_watches=%u child_watches=%u "
				     "capture_set=%u next_capture_frame=%u discovery_windows=%u\n", frame,
				     m_panel_source_count, m_upstream_source_count, m_producer_capture.index,
				     m_producer_capture.index < upstream.capture_frames.size()
				         ? upstream.capture_frames[m_producer_capture.index] : 0u, m_upstream_discovery_count);
			}
		}
	}
	if (match.unattributed_writes && ++m_producer_unattributed == 1u && m_producer_window) {
		const auto debug = buffer.GetDebugSnapshot();
		LOGF("PanelProducer coverage: frame=%u submit=%" PRIu64 " op=%u unattributed-address-or-unbound-buffer-writes\n",
		     frame, debug.submit_id, debug.op);
	}
	if (!match.Any()) return false;
	if ((match.images & PanelUpstreamWatchMask) != 0) ++m_producer_upstream_native_matches;
	++m_producer_order;
	++m_producer_matches;
	if (!m_producer_window || m_producer_emitted >= 64u) return false;
	++m_producer_emitted;
	return true;
}

void RenderExecutor::TracePanelProducerDraw(uint64_t submit_id, CommandBuffer& buffer,
	    const DrawCallInfo& draw, const DrawEmitInfo& emit, const DrawRenderState& state,
	    const RenderState& rendering, std::span<PreparedBindings* const> stages, uint32_t scalar_capture) {
	if (!PanelProducerTraceEnabled() || m_panel_source_count == 0) return;
	const auto frame = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
	if (frame > 4096u) return;
	const auto colors = std::span {state.color_info, state.color_count};
	auto match = MatchPanelProducerWrites(colors, stages);
	const auto& upstream = GetPanelUpstreamOptions();
	// Only an exact native write to a primary watch can teach one upstream level.
	// Observe every such draw before sparse trace/capture selection, never child writes.
	if (upstream.enabled && (match.images & PanelPrimaryWatchMask) != 0) {
		ObservePanelUpstreamSources(buffer, stages, match.images & PanelPrimaryWatchMask);
		// Replacing a child slot must not leave this record with its previous occupant's mask.
		match = MatchPanelProducerWrites(colors, stages);
	}
	const bool trace_due = TracePanelProducerWindow(buffer, frame, match) || scalar_capture != 0u;
	if (scalar_capture != 0u) {
		LOGF("PanelProducer GPU scalar link: call=%u sequence=%u frame=%u submit=%" PRIu64
		     " source_images=%x timing=before-this-draw\n",
		     match.Any() ? m_producer_order : 0u, scalar_capture, frame, submit_id, match.images);
	}
	static const std::string prefix = [] {
		const char* value = std::getenv("KYTY_VR_PANEL_PRODUCER_CAPTURE_PREFIX");
		return value != nullptr && std::strlen(value) <= 2048 ? std::string(value) : std::string {};
	}();
	const std::array<uint32_t, 2> milestones = upstream.enabled
	    ? upstream.capture_frames : std::array<uint32_t, 2> {128u, 1024u};
	const auto& capture_filter = GetPanelUpstreamCaptureFilter();
	const bool capture_parent = upstream.enabled && capture_filter.exact && upstream.parent_ps_hash != 0 &&
	                            capture_filter.ps_hash == upstream.parent_ps_hash;
	const uint32_t capture_images = upstream.enabled
	    ? match.images & (capture_parent ? PanelPrimaryWatchMask : PanelUpstreamWatchMask) : match.images;
	// Reuse the same pending sets and byte budget. Clears/no-input/ineligible draws do
	// not consume a set. Only an exact configured parent filter selects primary writers.
	const bool capture_due = capture_filter.enabled && capture_images != 0 && !prefix.empty() &&
	                         m_producer_capture.index < milestones.size() &&
	                         frame >= milestones[m_producer_capture.index];
	if (!trace_due && !capture_due) return;
	const auto& hw = buffer.GetRegisters();
	const auto& vp = hw.GetScreenViewport();
	const auto& viewport = vp.viewports[0];
	const auto scissor = calc_final_scissor(vp, hw.GetScanModeControl(), {rendering.width, rendering.height}, 0);
	const auto& mode = hw.GetModeControl();
	const auto& depth = state.depth_info;
	const bool clear = rendering.depth_stencil_attachment.depth_clear || rendering.depth_stencil_attachment.stencil_clear ||
	    std::any_of(rendering.color_attachments.begin(), rendering.color_attachments.end(),
	                [](const auto& attachment) { return attachment.is_clear; });
	const auto clock_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	std::string record = fmt::format("PanelProducer call={} kind=graphics frame={} submit={} tick={} clock_ms={} "
	    "source_images={:x} buffer_aliases={:x} metadata_aliases={:x} unattributed={} draw={} primitive={} count={} instances={} "
	    "vertex_offset={} first_vertex={} first_instance={} target_mask={:x} shader_mask={:x} color_mode={} color_op={:x} "
	    "rendering_clear={} ps_active={} es={:016x} gs={:016x} ps={:016x}\n",
	    m_producer_order, frame, submit_id, m_context.GetCommandScheduler().CurrentTick(), clock_ms,
	    match.images, match.buffer_aliases, match.metadata_aliases, match.unattributed_writes, draw.Name(),
	    static_cast<uint32_t>(buffer.GetUserConfig().GetPrimType()), draw.index_count, draw.instance_count,
	    emit.vertex_offset, emit.first_vertex, emit.first_instance, hw.GetRenderTargetMask(), hw.GetShaderRegisters().m_cbShaderMask,
	    hw.GetColorControl().mode, hw.GetColorControl().op, clear, state.ps_active,
	    buffer.GetShaders().GetVs().es_regs.data_addr, buffer.GetShaders().GetVs().gs_regs.data_addr, buffer.GetShaders().GetPs().ps_regs.data_addr);
	if (upstream.enabled) {
		record += fmt::format("PanelProducer ancestry call={} primary_images={:x} upstream_images={:x} "
		    "capture_depth=1 parent_ps={:016x}\n", m_producer_order,
		    match.images & PanelPrimaryWatchMask, match.images & PanelUpstreamWatchMask, upstream.parent_ps_hash);
	}
	if (capture_due) {
		record += fmt::format("PanelProducer capture_scope call={} capture_scope={} selected_depth={} "
		    "depth_kind=writer ancestry_trace_depth={} capture_images={:x}\n", m_producer_order,
		    upstream.enabled ? (capture_parent ? "exact-parent-writer" : "upstream-writer") : "legacy-watch-writer",
		    upstream.enabled ? (capture_parent ? "0" : "1") : "untracked", upstream.enabled ? 1u : 0u, capture_images);
	}
	record += fmt::format("PanelProducer coverage call={} viewport0_scale={},{},{} offset={},{},{} scissor0={},{},{},{} "
	    "cull={}/{} depth_test={} depth_write={} depth_compare={} stencil_test={} clip_disable={}\n",
	    m_producer_order, viewport.xscale, viewport.yscale, viewport.zscale, viewport.xoffset, viewport.yoffset, viewport.zoffset,
	    scissor.left, scissor.top, scissor.right, scissor.bottom, mode.cull_front, mode.cull_back, depth.depth_test_enable,
	    depth.depth_write_enable, static_cast<uint32_t>(depth.depth_compare_op), depth.stencil_test_enable, hw.GetClipControl().clip_disable);
	const auto& pa_stereo = hw.GetStereoControl();
	const auto& pa_stereo_x = hw.GetStereoX();
	const auto& ge_stereo = buffer.GetUserConfig().GetStereoControl();
	record += fmt::format("PanelProducer stereo call={} pa_stereo={:08x}/{} pa_stereo_x={:08x}/{} "
	    "ge_stereo={:08x}/{} raw_only=1\n", m_producer_order,
	    pa_stereo.value, pa_stereo.observed, pa_stereo_x.value, pa_stereo_x.observed,
	    ge_stereo.value, ge_stereo.observed);
	for (const auto& color: colors) {
		const auto& blend = hw.GetBlendControl(color.target_slot);
		const auto& metadata = color.desc.info.metadata;
		record += fmt::format("PanelProducer color call={} slot={} blend={} rgb={}/{}/{} alpha={}/{}/{} separate_alpha={} "
		    "metadata_kind={} metadata={:016x}:{} declared_clear_word={:08x} materialization=unobserved\n",
		    m_producer_order, color.target_slot, blend.enable, blend.color_srcblend, blend.color_comb_fcn, blend.color_destblend,
		    blend.alpha_srcblend, blend.alpha_comb_fcn, blend.alpha_destblend, blend.separate_alpha_blend,
		    static_cast<uint32_t>(metadata.kind), metadata.range.address, metadata.range.size, metadata.clear_word);
	}
	if (trace_due && match.images != 0 && PanelNativeScalarTraceEnabled()) {
		// Resolve through RenderExecutor's existing read-only cache access. The free
		// helper receives stable const metadata only; GetImage would touch the cache.
		std::array<const ImageInfo*, 9> attachment_infos {};
		size_t attachment_count = 0;
		bool invalid_attachment = false;
		const auto append_attachment = [&](ImageId id) {
			if (attachment_count == attachment_infos.size()) {
				invalid_attachment = true;
				return;
			}
			const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
			invalid_attachment |= image == nullptr;
			attachment_infos[attachment_count++] = image != nullptr ? &image->info : nullptr;
		};
		for (const auto& color: colors) append_attachment(color.image_id);
		if (depth.image_id) append_attachment(depth.image_id);
		AppendPanelNativeScalars(record, m_producer_order, buffer, colors,
		                        std::span {attachment_infos.data(), attachment_count}, invalid_attachment, stages);
	}
	TracePanelBindings("PanelProducer", m_producer_order, buffer, std::move(record), colors, depth.image_id, stages,
	                   clear, trace_due, capture_due ? &prefix : nullptr,
	                   capture_due ? milestones[m_producer_capture.index] : 0u, m_producer_capture);
}

void RenderExecutor::TracePanelProducerDispatch(uint64_t submit_id, CommandBuffer& buffer, PreparedBindings& bindings,
	    std::array<uint32_t, 3> groups, uint32_t mode, bool async_compute, vk::Buffer indirect_buffer, uint64_t indirect_offset) {
	if (!PanelProducerTraceEnabled() || m_panel_source_count == 0) return;
	const auto frame = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
	if (frame > 4096u) return;
	PreparedBindings* stage = &bindings;
	const auto stages = std::span {&stage, 1u};
	const auto match = MatchPanelProducerWrites({}, stages);
	if (!TracePanelProducerWindow(buffer, frame, match)) return;
	const auto& program = *bindings.runtime->program;
	std::string record = fmt::format("PanelProducer call={} kind=compute frame={} submit={} tick={} source_images={:x} "
	    "buffer_aliases={:x} metadata_aliases={:x} unattributed={} async={} groups_known={} groups={},{},{} "
	    "mode={:x} indirect_buffer={:016x} indirect_offset={} cs={:016x} hash={:016x}\n",
	    m_producer_order, frame, submit_id, m_context.GetCommandScheduler().CurrentTick(),
	    match.images, match.buffer_aliases, match.metadata_aliases, match.unattributed_writes, async_compute,
	    indirect_buffer == nullptr, groups[0], groups[1], groups[2], mode, PanelNativeHandle(indirect_buffer), indirect_offset,
	    buffer.GetShaders().GetCs().cs_regs.data_addr, program.shader_hash);
	TracePanelBindings("PanelProducer", m_producer_order, buffer, std::move(record), {}, {}, stages,
	                   false, true, nullptr, 0, m_producer_capture);
}

void RenderExecutor::TracePanelProducerShortcut(const CommandBuffer& buffer, const char* operation, GuestRange range,
	    uint64_t shader_hash, uint32_t value, bool metadata, ImageId image, const ImageViewInfo* view) {
	if (!PanelProducerTraceEnabled() || m_panel_source_count == 0) return;
	const auto frame = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
	if (frame > 4096u) return;
	PanelProducerMatch match {};
	if (image && view != nullptr) match.images = MatchPanelSourceImage(image, *view);
	if (metadata) match.metadata_aliases = MatchPanelSourceRange(range, true);
	else match.buffer_aliases = MatchPanelSourceRange(range, false);
	// A successful ClearImageFromBuffer selects exactly one native image for this
	// complete guest range. A mere overlapping buffer write is kept as an alias above.
	if (std::strcmp(operation, "compute-buffer-clear") == 0) {
		for (uint32_t i = 0; i < PanelSourceLimit(); ++i) {
			const auto& watch = m_panel_sources[i];
			if ((match.buffer_aliases & (1u << i)) != 0 && watch.data == range)
				match.images |= MatchPanelSourceImage(watch.image_id, watch.view);
		}
	}
	if (!TracePanelProducerWindow(buffer, frame, match)) return;
	const auto debug = buffer.GetDebugSnapshot();
	LOGF("PanelProducer shortcut: call=%u frame=%u submit=%" PRIu64 " tick=%" PRIu64
	     " operation=%s source_images=0x%x buffer_aliases=0x%x metadata_aliases=0x%x "
	     "guest=0x%016" PRIx64 " bytes=%" PRIu64 " hash=0x%016" PRIx64 " value=0x%08x op=%u\n",
	     m_producer_order, frame, debug.submit_id,
	     m_context.GetCommandScheduler().CurrentTick(), operation, match.images, match.buffer_aliases,
	     match.metadata_aliases, range.address, range.size, shader_hash, value, debug.op);
}

void RenderExecutor::TracePanelDraw(uint64_t submit_id, CommandBuffer& buffer,
                                    const DrawCallInfo& draw, const DrawEmitInfo& emit,
                                    const DrawRenderState& state,
                                    const RenderState& rendering,
                                    std::span<PreparedBindings* const> stages) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_VR_PANEL_DRAW_TRACE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled || m_panel_trace_draws >= 16384u) return;
	auto& cache = m_context.GetTextureCache();
	uint32_t presented_mask = 0;
	for (uint32_t i = 0; i < state.color_count; ++i) {
		const auto* image = cache.m_slot_images.try_get(state.color_info[i].image_id);
		if (image != nullptr && image->usage.video_out) presented_mask |= 1u << i;
	}
	if (presented_mask == 0) return;
	ObservePanelSources(buffer, stages);
	const auto call = ++m_panel_trace_draws;
	static const std::string capture_prefix = [] {
		const char* value = std::getenv("KYTY_VR_PANEL_DRAW_CAPTURE_PREFIX");
		return value != nullptr && std::strlen(value) <= 2048 ? std::string(value) : std::string {};
	}();
	constexpr std::array<uint32_t, 2> capture_milestones {128u, 1024u};
	const bool capture_due = !capture_prefix.empty() && m_panel_capture.index < capture_milestones.size() &&
	                         call >= capture_milestones[m_panel_capture.index];
	const bool trace_due = call <= 8u || (call & 255u) <= 1u;
	if (!trace_due && !capture_due) return;
	// Download ends dynamic rendering. Never split a pass whose clear load operation
	// could be replayed when an otherwise identical following draw begins rendering.
	const bool rendering_clear = rendering.depth_stencil_attachment.depth_clear ||
	    rendering.depth_stencil_attachment.stencil_clear ||
	    std::any_of(rendering.color_attachments.begin(), rendering.color_attachments.end(),
	                [](const auto& attachment) { return attachment.is_clear; });
	const auto& hw = buffer.GetRegisters();
	const auto& sh = hw.GetShaderRegisters();
	const auto clock_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	    std::chrono::steady_clock::now().time_since_epoch()).count());
	std::string record = fmt::format(
	    "PanelDraw call={} submit={} frame={} tick={} clock_ms={} draw={} primitive={} count={} instances={} "
	    "vertex_offset={} first_vertex={} first_instance={} presented_mask={:x} target_mask={:x} "
	    "shader_mask={:x} color_mode={} rendering_clear={} es={:016x} gs={:016x} ps={:016x}\n",
	    call, submit_id, m_context.GetGpu().GetFrameNum(), m_context.GetCommandScheduler().CurrentTick(),
	    clock_ms, draw.Name(),
	    static_cast<uint32_t>(buffer.GetUserConfig().GetPrimType()), draw.index_count, draw.instance_count,
	    emit.vertex_offset, emit.first_vertex, emit.first_instance, presented_mask, hw.GetRenderTargetMask(),
	    sh.m_cbShaderMask, hw.GetColorControl().mode, rendering_clear,
	    buffer.GetShaders().GetVs().es_regs.data_addr,
	    buffer.GetShaders().GetVs().gs_regs.data_addr, buffer.GetShaders().GetPs().ps_regs.data_addr);
	TracePanelBindings("PanelDraw", call, buffer, std::move(record),
	                   std::span {state.color_info, state.color_count}, state.depth_info.image_id, stages,
	                   rendering_clear, trace_due, capture_due ? &capture_prefix : nullptr,
	                   capture_due ? capture_milestones[m_panel_capture.index] : 0u, m_panel_capture);
}

void RenderExecutor::TracePanelBindings(const char* label, uint32_t call, CommandBuffer& buffer,
	                                      std::string record, std::span<const RenderColorInfo> colors,
	                                      ImageId depth_id, std::span<PreparedBindings* const> stages,
	                                      bool rendering_clear, bool trace_due, const std::string* capture_prefix,
	                                      uint32_t milestone, PanelCaptureState& capture) {
	auto& cache = m_context.GetTextureCache();
	const auto* capture_filter = std::strcmp(label, "PanelProducer") == 0 ? &GetPanelUpstreamCaptureFilter() : nullptr;
	const bool exact_capture = capture_filter != nullptr && capture_filter->exact;
	const bool capture_due = capture_prefix != nullptr && (capture_filter == nullptr || capture_filter->enabled);
	const bool vertex_multiplier_capture = capture_due && capture_filter != nullptr &&
	                                       PanelVertexMultiplierCaptureEnabled();
	// video_out marks the exact cache image selected by the presenter. Explicit preview
	// paths can mark other images too; addresses/IDs correlate with the surface trace.
	for (uint32_t i = 0; i < colors.size(); ++i) {
		const auto& target = colors[i];
		const auto* image = cache.m_slot_images.try_get(target.image_id);
		if (image == nullptr) continue;
		const auto& view = target.desc.view_info;
		record += fmt::format(
		    "{} target call={} slot={} presented={} id={} native={:016x} guest={:016x} "
		    "cached={:016x} guest_fmt={} native_fmt={} view_fmt={} extent={}x{} native_extent={}x{} "
		    "mip={}:{} layer={}:{} guest_mip={} guest_layer={}\n", label,
		    call, target.target_slot, image->usage.video_out, target.image_id.index,
		    PanelNativeHandle(image->backing.image), target.desc.info.data.address, image->info.data.address,
		    static_cast<uint32_t>(target.desc.info.guest_format), static_cast<uint32_t>(image->backing.format),
		    static_cast<uint32_t>(view.format), target.Extent().width, target.Extent().height,
		    image->backing.extent.width, image->backing.extent.height, view.base_level, view.level_count,
		    view.base_layer, view.layer_count, target.guest_mip_level, target.guest_array_layer);
	}
	struct Candidate {
		Image* image;
		const TextureBinding* binding;
		const ImageViewInfo* view;
		const ShaderRecompiler::IR::ImageResource* resource;
		uint32_t stage;
		uint32_t index;
		uint64_t shader_hash;
		bool vertex_multiplier;
	};
	std::array<Candidate, 16> candidates {};
	uint32_t candidate_count = 0;
	uint32_t image_lines = 0;
	bool uncertain_writes = false;
	for (uint32_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
		const auto& prepared = *stages[stage_index];
		const auto& program = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		uncertain_writes |= program.has_address_writes ||
		    std::any_of(program.info.buffers.begin(), program.info.buffers.end(),
		                [](const auto& resource) { return resource.written; });
		record += fmt::format("{} stage call={} index={} stage={} hash={:016x} images={} "
		                      "samplers={} pairs={} address_writes={}\n", label, call, stage_index,
		                      static_cast<uint32_t>(program.stage), program.shader_hash,
		                      prepared.images.size(), program.info.samplers.size(),
		                      program.info.sampled_pairs.size(), program.has_address_writes);

		if (std::strcmp(label, "PanelProducer") == 0) {
			const auto append_words = [&](const char* kind, std::span<const uint32_t> words) {
				const auto shown = std::min<size_t>(words.size(), 64u);
				record += fmt::format("{} values call={} stage={} kind={} base_reg={} shown={} total={} words=",
				                      label, call, stage_index, kind, program.user_data_base, shown, words.size());
				for (size_t i = 0; i < shown; ++i) record += fmt::format("{:08x},", words[i]);
				record += "\n";
			};
			// These are immutable host snapshots or already-uploaded shader data. Never read
			// a guest buffer or wait for GPU contents just to inspect a possible fade constant.
			append_words("snapshot-user-data", snapshot.user_data);
			append_words("user-register-map", program.info.user_data_registers);
			append_words("prepared-shader-data", prepared.shader_data);
			append_words("snapshot-flattened-srt", snapshot.flattened_srt);
			const auto buffers = std::min({program.info.buffers.size(), snapshot.buffers.size(), size_t {16}});
			for (uint32_t i = 0; i < buffers; ++i) {
				const auto& resource = program.info.buffers[i];
				const auto& words = snapshot.buffers[i];
				ShaderBufferResource descriptor {};
				std::memcpy(descriptor.fields, words.dwords.data(), sizeof(descriptor.fields));
				const auto index = resource.descriptor_index;
				const auto* source = index < prepared.buffer_sources.size() ? &prepared.buffer_sources[index] : nullptr;
				const auto* native = index < prepared.buffers.size() ? &prepared.buffers[index] : nullptr;
				record += fmt::format("{} buffer call={} stage={} resource={} binding={} read={} written={} "
				                      "scalar={} image_alias={} first_pc={:x} words={} sharp={:08x},{:08x},{:08x},{:08x} "
				                      "guest={:016x} size={} resolved={:016x}:{} native={:016x} offset={} range={}\n",
				                      label, call, stage_index, i, index, resource.read, resource.written,
				                      resource.scalar, resource.image_alias, resource.first_use_pc, words.dword_count,
				                      words.dwords[0], words.dwords[1], words.dwords[2], words.dwords[3],
				                      descriptor.Base48(), descriptor.GetSize(), source ? source->address : 0,
				                      source ? source->size : 0, native ? PanelNativeHandle(native->buffer) : 0,
				                      native ? native->offset : 0, native ? native->range : 0);
			}
			if (buffers != program.info.buffers.size())
				record += fmt::format("{} truncated call={} stage={} shown_buffers={} total_buffers={}\n",
				                      label, call, stage_index, buffers, program.info.buffers.size());
		}
		const auto count = std::min({prepared.images.size(), program.info.images.size(), snapshot.images.size()});
		uint32_t shown = 0;
		for (uint32_t i = 0; i < count && image_lines < candidates.size(); ++i, ++image_lines, ++shown) {
			const auto& binding = prepared.images[i];
			const auto& resource = program.info.images[i];
			const auto& value = snapshot.images[i];
			const auto& words = value.dwords;
			ShaderTextureResource descriptor {};
			std::memcpy(descriptor.fields, words.data(), sizeof(descriptor.fields));
			record += fmt::format(
			    "{} descriptor call={} stage={} image={} binding={} class={} read={} written={} "
			    "first_pc={:x} r128={} words={} tsharp={:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x} "
			    "address={:016x} format={} extent={}x{} tile={} swizzle={:03x}\n", label,
			    call, stage_index, i, resource.descriptor_index, static_cast<uint32_t>(resource.resource_class),
			    resource.read, resource.written, resource.first_use_pc, resource.r128, value.dword_count,
			    words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
			    descriptor.Base40(), static_cast<uint32_t>(descriptor.Format()), descriptor.Width5() + 1u,
			    descriptor.Height5() + 1u, static_cast<uint32_t>(descriptor.TileMode()), descriptor.DstSelXYZW());
			auto* image = cache.m_slot_images.try_get(binding.image_id);
			if (image == nullptr) {
				record += fmt::format("{} input call={} stage={} image={} state=missing-cache-image\n", label,
				                      call, stage_index, i);
				continue;
			}
			const auto found = std::find_if(image->views.begin(), image->views.end(),
			    [&](const auto& native) { return native.view == binding.image_view; });
			const auto* actual_view = found == image->views.end() ? nullptr : &found->info;
			const auto& view = actual_view != nullptr ? *actual_view : binding.desc.view_info;
			record += fmt::format(
			    "{} input call={} stage={} image={} id={} native={:016x} view={:016x} "
			    "actual_view={} requested={:016x} cached={:016x} native_fmt={} requested_view_fmt={} view_fmt={} "
			    "native_extent={}x{}x{} samples={} levels={} layers={} mip={}:{} layer={}:{} aspect={:x} "
			    "mapping={},{},{},{} layout={} dirty={}/{}/{} dynamic_views={}\n", label,
			    call, stage_index, i, binding.image_id.index, PanelNativeHandle(image->backing.image),
			    PanelNativeHandle(binding.image_view), actual_view != nullptr, binding.desc.info.data.address,
			    image->info.data.address, static_cast<uint32_t>(image->backing.format),
			    static_cast<uint32_t>(binding.desc.view_info.format), static_cast<uint32_t>(view.format),
			    image->backing.extent.width, image->backing.extent.height, image->backing.extent.depth,
			    image->backing.samples, image->backing.mip_levels, image->backing.layers,
			    view.base_level, view.level_count, view.base_layer, view.layer_count,
			    static_cast<uint32_t>(view.aspect), static_cast<uint32_t>(view.mapping.r),
			    static_cast<uint32_t>(view.mapping.g), static_cast<uint32_t>(view.mapping.b),
			    static_cast<uint32_t>(view.mapping.a), static_cast<uint32_t>(binding.layout),
			    image->IsCpuDirty(), image->IsBufferModified(), image->IsGpuModified(), binding.mip_views.size());
			// Trace every binding as before; selection uses this actual program and its
			// logical image index, never a descriptor-kind bank slot.
			const bool multiplier_input = vertex_multiplier_capture && program.stage == ShaderType::Vertex &&
			                              program.shader_hash == PanelMultiplierVertexShader && i == 0u;
			if (!exact_capture || (program.stage == ShaderType::Pixel && program.shader_hash == capture_filter->ps_hash &&
			                       i == capture_filter->image_index) || multiplier_input) {
				candidates[candidate_count++] = {image, &binding, actual_view, &resource, stage_index, i,
				                                 program.shader_hash, multiplier_input};
			}
		}
		if (shown != prepared.images.size()) {
			record += fmt::format("{} truncated call={} stage={} shown_images={} total_images={}\n", label,
			                      call, stage_index, shown, prepared.images.size());
		}
		const auto pair_count = std::min<size_t>(program.info.sampled_pairs.size(), 8u);
		for (uint32_t i = 0; i < pair_count; ++i) {
			const auto& pair = program.info.sampled_pairs[i];
			if (pair.sampler >= program.info.samplers.size()) continue;
			const auto index = program.info.samplers[pair.sampler].snapshot_index;
			if (index >= snapshot.samplers.size()) continue;
			const auto& value = snapshot.samplers[index];
			record += fmt::format("{} pair call={} stage={} image={} sampler={} first_pc={:x} "
			                      "words={} sampler_raw={:08x},{:08x},{:08x},{:08x}\n", label, call,
			                      stage_index, pair.image, pair.sampler, pair.first_use_pc, value.dword_count,
			                      value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3]);
		}
		if (pair_count != program.info.sampled_pairs.size()) {
			record += fmt::format("{} truncated call={} stage={} shown_pairs={} total_pairs={}\n", label,
			                      call, stage_index, pair_count, program.info.sampled_pairs.size());
		}
	}
	if (trace_due) LOGF("%s", record.c_str());
	if (!capture_due) return;
	if (vertex_multiplier_capture) {
		// Vertex bindings are traced first, but the companion must never consume a set
		// unless the selected pixel input has already queued successfully.
		std::stable_partition(candidates.begin(), candidates.begin() + candidate_count,
		    [](const Candidate& candidate) { return !candidate.vertex_multiplier; });
	}
	constexpr uint64_t set_limit = 64u * 1024u * 1024u;
	constexpr uint64_t total_limit = 2u * set_limit;
	uint64_t set_bytes = 0;
	std::array<vk::Image, 2> captured {};
	uint32_t captured_count = 0;
	bool primary_queued = false;
	bool multiplier_queued = false;
	const char* multiplier_reason = "no-resolved-exact-vertex-input";
	const char* reason = rendering_clear ? "rendering-clear-would-be-replayed"
	                   : uncertain_writes ? "shader-buffer-or-address-writes"
	                   : exact_capture ? "capture-filter-no-resolved-pixel-input" : "no-sampled-input";
	for (uint32_t i = 0; !rendering_clear && !uncertain_writes && i < candidate_count &&
	                       captured_count < captured.size(); ++i) {
		const auto& candidate = candidates[i];
		if (candidate.vertex_multiplier && !primary_queued) {
			multiplier_reason = "primary-lut-not-queued";
			continue;
		}
		const char*& candidate_reason = candidate.vertex_multiplier ? multiplier_reason : reason;
		const auto& resource = *candidate.resource;
		auto& image = *candidate.image;
		const auto& backing = image.backing;
		const auto* view = candidate.view;
		const bool plain_2d = view != nullptr && view->type == vk::ImageViewType::e2D &&
		                      backing.image_type == vk::ImageType::e2D && backing.extent.depth == 1;
		// Whole-volume readback is available only for an explicitly selected producer
		// input. A 3D image has one Vulkan array layer; Z slices belong to its extent.
		const bool exact_volume = exact_capture && view != nullptr && view->type == vk::ImageViewType::e3D &&
		                          backing.image_type == vk::ImageType::e3D && image.info.IsVolume() &&
		                          image.info.extent == backing.extent && backing.extent.depth != 0 &&
		                          backing.extent.depth <= 256;
		if (resource.resource_class != ShaderRecompiler::IR::ImageResourceClass::Sampled ||
		    !resource.read || resource.written || candidate.binding->desc.type != TextureCache::BindingType::Texture) {
			if (exact_capture) candidate_reason = "selected-input-is-not-read-only-sampled-texture";
			continue;
		}
		if (view == nullptr || view->aspect != vk::ImageAspectFlagBits::eColor ||
		    (!plain_2d && !exact_volume) || view->base_level != 0 || view->level_count != 1 ||
		    view->base_layer != 0 || view->layer_count != 1 || candidate.binding->mip_views.size() > 1 ||
		    backing.image == nullptr || !image.registered || image.depth_id || image.info.IsDepth() ||
		    backing.samples != 1 || backing.mip_levels != 1 ||
		    backing.layers != 1 || image.info.resources.levels != 1 || image.info.resources.layers != 1 ||
		    !backing.subresource_states.empty() || backing.state.layout == vk::ImageLayout::eUndefined ||
		    backing.state.layout == vk::ImageLayout::ePreinitialized ||
		    !(backing.usage & vk::ImageUsageFlagBits::eTransferSrc) ||
		    backing.extent.width == 0 || backing.extent.height == 0 ||
		    backing.extent.width > 8192 || backing.extent.height > 8192) {
			candidate_reason = "unsupported-native-image-or-view";
			continue;
		}
		if (candidate.vertex_multiplier &&
		    (!plain_2d || backing.format != vk::Format::eR32G32B32A32Sfloat ||
		     view->format != vk::Format::eR32G32B32A32Sfloat ||
		     backing.extent.width != 1 || backing.extent.height != 1 || backing.extent.depth != 1)) {
			candidate_reason = "vertex-multiplier-requires-1x1-rgba32f-2d-view";
			continue;
		}
		if (std::find(captured.begin(), captured.end(), backing.image) != captured.end()) {
			if (candidate.vertex_multiplier) multiplier_reason = "input-already-captured";
			continue;
		}
		bool overlaps = false;
		for (uint32_t target = 0; target < colors.size(); ++target) {
			const auto* other = cache.m_slot_images.try_get(colors[target].image_id);
			overlaps |= other != nullptr && PanelImagesOverlap(image, *other);
		}
		if (const auto* depth = cache.m_slot_images.try_get(depth_id); depth != nullptr) {
			overlaps |= PanelImagesOverlap(image, *depth);
		}
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.desc.type != TextureCache::BindingType::Storage) continue;
				const auto* other = cache.m_slot_images.try_get(binding.image_id);
				overlaps |= other != nullptr && PanelImagesOverlap(image, *other);
			}
		}
		if (overlaps) {
			candidate_reason = "input-overlaps-render-or-storage-write";
			continue;
		}
		const auto texel_bytes = PanelCaptureTexelBytes(backing.format);
		const uint64_t bytes = static_cast<uint64_t>(backing.extent.width) * backing.extent.height *
		                       backing.extent.depth * texel_bytes;
		if (bytes == 0 || bytes > set_limit || bytes > set_limit - set_bytes ||
		    bytes > total_limit - capture.bytes) {
			candidate_reason = "unsupported-native-format-or-byte-cap";
			continue;
		}
		const auto path = fmt::format("{}-draw{}-stage{}-image{}-{:016x}.bin", *capture_prefix,
		                              call, candidate.stage, candidate.index, PanelNativeHandle(backing.image));
		auto metadata = record + fmt::format("selected stage={} image={} milestone={}\n",
		    candidate.stage, candidate.index, milestone);
		if (candidate.vertex_multiplier) {
			metadata += fmt::format("selected_filter mode=exact-vertex-multiplier primary_ps={:016x} "
			    "primary_logical_image=3 requested_vs={:016x} requested_logical_image=0 actual_vs={:016x} "
			    "actual_logical_image={} descriptor_binding={} primary_queued=1 index_scope=program.info.images\n",
			    PanelMultiplierPrimaryPs, PanelMultiplierVertexShader, candidate.shader_hash,
			    candidate.index, resource.descriptor_index);
		} else if (exact_capture) {
			metadata += fmt::format("selected_filter mode=exact-pixel-input requested_ps={:016x} "
			    "requested_logical_image={} actual_ps={:016x} actual_logical_image={} descriptor_binding={} "
			    "index_scope=program.info.images\n", capture_filter->ps_hash, capture_filter->image_index,
			    candidate.shader_hash, candidate.index, resource.descriptor_index);
		}
		if (!QueuePanelInputCapture(buffer, image, static_cast<uint32_t>(bytes), path, metadata, label)) {
			candidate_reason = "staging-allocation-failed";
			continue;
		}
		if (candidate.vertex_multiplier) multiplier_queued = true;
		else if (vertex_multiplier_capture) primary_queued = true;
		captured[captured_count++] = backing.image;
		set_bytes += bytes;
		capture.bytes += bytes;
	}
	if (vertex_multiplier_capture && primary_queued && !multiplier_queued) {
		LOGF("%s capture companion skipped: call=%u milestone=%u mode=exact-vertex-multiplier "
		     "reason=%s primary_queued=1\n", label, call, milestone, multiplier_reason);
	}
	if (captured_count != 0) {
		if (!trace_due) LOGF("%s", record.c_str());
		LOGF("%s capture set: call=%u milestone=%u inputs=%u bytes=%" PRIu64
		     " total_bytes=%" PRIu64 " input_cap=2 set_cap=%" PRIu64 "\n", label, call,
		     milestone, captured_count, set_bytes, capture.bytes, set_limit);
		++capture.index;
		capture.misses = 0;
	} else if (++capture.misses <= 4u || ((capture.misses & 255u) == 0 && capture.misses <= 16384u)) {
		LOGF("%s capture deferred: call=%u milestone=%u candidates=%u reason=%s misses=%u\n",
		     label, call, milestone, candidate_count, reason, capture.misses);
	}
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	auto& ucfg = buffer.GetUserConfig();
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
		descriptor_stages[stage_count++] = &*bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	auto& pipeline = m_context.GetPipelineCache().GetGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.programs);
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
	                         feedback_aspects, stages);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	const auto scalar_capture = CapturePanelGpuScalarBeforeDraw(
	    submit_id, buffer, draw, emit, state, rendering, stages);
	if (mesh_active) {
		const uint32_t draw_data[] {
		    draw.index_count,
		    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
		    emit.first_instance, index_source.guest_element_size,
		    static_cast<uint32_t>(index_source.address),
		    static_cast<uint32_t>(index_source.address >> 32u)};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(draw_data), draw_data);
	} else {
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), state.depth_info,
	                           rendering);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);

	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	TracePanelDraw(submit_id, buffer, draw, emit, state, rendering, stages);
	TracePanelProducerDraw(submit_id, buffer, draw, emit, state, rendering, stages, scalar_capture);
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || args.instance_count == 0) {
		return;
	}

	const bool metadata_operation = ConsumeMetadataColorOperation(buffer);
	if (metadata_operation || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		if (metadata_operation && PanelProducerTraceEnabled()) {
			for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; ++slot) {
				const auto& target = buffer.GetRegisters().GetRenderTarget(slot);
				if (target.base.addr != 0) TracePanelProducerShortcut(buffer, "metadata-color-consumed-no-native-draw",
				    {target.base.addr, 1}, 0, buffer.GetRegisters().GetColorControl().mode);
			}
		}
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
	const bool native_indirect = indirect && !state.vertex_info[0].fetch_embedded;
	emit.vertex_offset  = native_indirect ? 0 : vertex_offset + args.base_vertex;
	emit.first_instance = native_indirect ? 0 : instance_offset;

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	if (args.vertex_count == 0 || args.instance_count == 0) {
		return;
	}

	const auto& copy_control = buffer.GetRegisters().GetRenderControl();
	if (copy_control.copy_depth_to_color || copy_control.copy_stencil_to_color) {
		const char* enabled = std::getenv("KYTY_VR_DEPTH_STENCIL_COLOR_COPY");
		if (enabled != nullptr && std::strcmp(enabled, "1") == 0 && DepthStencilColorCopy(buffer, args)) {
			ResetBindings();
			return;
		}
	}

	const bool metadata_operation = ConsumeMetadataColorOperation(buffer);
	if (metadata_operation || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		if (metadata_operation && PanelProducerTraceEnabled()) {
			for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; ++slot) {
				const auto& target = buffer.GetRegisters().GetRenderTarget(slot);
				if (target.base.addr != 0) TracePanelProducerShortcut(buffer, "metadata-color-consumed-no-native-draw",
				    {target.base.addr, 1}, 0, buffer.GetRegisters().GetColorControl().mode);
			}
		}
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	const bool native_indirect = indirect && !state.vertex_info[0].fetch_embedded;
	emit.first_vertex = native_indirect ? 0 :
	    static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = native_indirect ? 0 : instance_offset;

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	auto resolved_view = dst.desc.view_info;
	resolved_view.base_level = dst.guest_mip_level;
	resolved_view.level_count = 1;
	resolved_view.base_layer = dst.guest_array_layer;
	resolved_view.layer_count = 1;
	TracePanelProducerShortcut(buffer, "color-resolve", dst.desc.info.data, 0, 0, false,
	                           dst.image_id, &resolved_view);
	return true;
}

} // namespace Libs::Graphics
