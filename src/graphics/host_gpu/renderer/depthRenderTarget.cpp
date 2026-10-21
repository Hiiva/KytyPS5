#include "graphics/host_gpu/renderer/depthRenderTarget.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
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
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <span>

namespace Libs::Graphics {

static bool DepthCopyTraceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_VR_TRACE_DEPTH_COPY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static void TraceDepthCopyState(const CommandBuffer& buffer) {
	if (!DepthCopyTraceEnabled()) {
		return;
	}
	const auto& hw = buffer.GetRegisters();
	const auto& z  = hw.GetDepthRenderTarget();
	const auto& rc = hw.GetRenderControl();
	const auto& dc = hw.GetDepthControl();
	const auto& cb = hw.GetRenderTarget(0);
	const auto& sh = hw.GetShaderRegisters();
	const auto ps = buffer.GetShaders().GetPs().ps_regs.data_addr;
	const auto& vs = buffer.GetShaders().GetVs();
	const auto& ucfg = buffer.GetUserConfig();
	const auto& vp = hw.GetScreenViewport();
	const auto& vp0 = vp.viewports[0];
	const auto& clip = hw.GetClipControl();
	const auto& mode = hw.GetModeControl();
	const auto& scan = hw.GetScanModeControl();
	const auto& masks = hw.GetAaMaskControl();
	const auto& blend = hw.GetBlendControl(0);
	const auto debug = buffer.GetDebugSnapshot();
	const auto scissor = calc_final_scissor(
	    vp, scan, {cb.attrib2.width + 1u, cb.attrib2.height + 1u}, 0);
	const uint64_t coverage_flags =
	    static_cast<uint64_t>(mode.cull_front) |
	    (static_cast<uint64_t>(mode.cull_back) << 1u) |
	    (static_cast<uint64_t>(mode.face) << 2u) |
	    (static_cast<uint64_t>(mode.poly_mode) << 3u) |
	    (static_cast<uint64_t>(mode.polymode_front_ptype) << 8u) |
	    (static_cast<uint64_t>(mode.polymode_back_ptype) << 12u) |
	    (static_cast<uint64_t>(mode.vtx_window_offset_enable) << 16u) |
	    (static_cast<uint64_t>(clip.clip_disable) << 17u) |
	    (static_cast<uint64_t>(clip.dx_clip_space) << 18u) |
	    (static_cast<uint64_t>(clip.min_z_clip_disable) << 19u) |
	    (static_cast<uint64_t>(clip.max_z_clip_disable) << 20u) |
	    (static_cast<uint64_t>(clip.force_viewport_index_from_vs_enable) << 21u) |
	    (static_cast<uint64_t>(scan.msaa_enable) << 22u) |
	    (static_cast<uint64_t>(scan.vport_scissor_enable) << 23u) |
	    (static_cast<uint64_t>(clip.user_clip_planes) << 24u) |
	    (static_cast<uint64_t>(clip.user_clip_plane_mode) << 32u);
	const uint64_t flags =
	    static_cast<uint64_t>(rc.copy_depth_to_color) |
	    (static_cast<uint64_t>(rc.copy_stencil_to_color) << 1u) |
	    (static_cast<uint64_t>(rc.copy_centroid) << 2u) |
	    (static_cast<uint64_t>(rc.copy_sample) << 3u) |
	    (static_cast<uint64_t>(rc.depth_clear_enable) << 7u) |
	    (static_cast<uint64_t>(rc.stencil_clear_enable) << 8u) |
	    (static_cast<uint64_t>(rc.depth_compress_disable) << 9u) |
	    (static_cast<uint64_t>(rc.stencil_compress_disable) << 10u) |
	    (static_cast<uint64_t>(rc.resummarize_enable) << 11u) |
	    (static_cast<uint64_t>(dc.z_enable) << 12u) |
	    (static_cast<uint64_t>(dc.z_write_enable) << 13u) |
	    (static_cast<uint64_t>(dc.stencil_enable) << 14u) |
	    (static_cast<uint64_t>(dc.depth_bounds_enable) << 15u) |
	    (static_cast<uint64_t>(z.depth_view.depth_write_disable) << 16u) |
	    (static_cast<uint64_t>(z.depth_view.stencil_write_disable) << 17u);
	const auto pair = [](uint32_t a, uint32_t b) { return (static_cast<uint64_t>(a) << 32u) | b; };
	auto key = std::to_array<uint64_t>({
	    flags,
	    pair(static_cast<uint32_t>(z.z_info.format), static_cast<uint32_t>(z.stencil_info.format)),
	    pair(z.size.x_max, z.size.y_max), pair(z.size.valid, z.z_info.num_samples),
	    pair(z.depth_view.slice_start, z.depth_view.slice_max), pair(dc.zfunc, z.z_info.htile_acceleration),
	    pair(static_cast<uint32_t>(cb.info.format), static_cast<uint32_t>(cb.info.channel_type)),
	    pair(static_cast<uint32_t>(cb.info.channel_order), static_cast<uint32_t>(cb.attrib3.tile_mode)),
	    pair(cb.attrib2.width, cb.attrib2.height),
	    pair(cb.view.base_array_slice_index, cb.view.last_array_slice_index),
	    pair(cb.attrib.num_samples, cb.attrib.num_fragments),
	    pair(hw.GetRenderTargetMask() & 0xfu, sh.m_cbShaderMask & 0xfu),
	    pair(sh.target_output_mode[0], hw.GetColorControl().mode), ps,
	    pair(z.z_read_base_addr == cb.base.addr, z.stencil_read_base_addr == cb.base.addr),
	    pair(z.depth_view.current_mip_level, cb.view.current_mip_level),
	    pair(cb.attrib3.dimension, cb.attrib3.depth),
	    pair(debug.op, static_cast<uint32_t>(ucfg.GetPrimType())),
	    pair(debug.args[0], debug.args[1]), pair(debug.args[2], debug.args[3]), debug.arg4,
	    vs.es_regs.data_addr, vs.gs_regs.data_addr, ucfg.GetIndexOffset(),
	    pair(std::bit_cast<uint32_t>(vp0.xscale), std::bit_cast<uint32_t>(vp0.xoffset)),
	    pair(std::bit_cast<uint32_t>(vp0.yscale), std::bit_cast<uint32_t>(vp0.yoffset)),
	    pair(std::bit_cast<uint32_t>(vp0.zscale), std::bit_cast<uint32_t>(vp0.zoffset)),
	    pair(scissor.left, scissor.top), pair(scissor.right, scissor.bottom),
	    pair(vp.transform_control, vp.clip_rect_rule),
	    pair(vp.window_offset_x, vp.window_offset_y), pair(vp.hw_offset_x, vp.hw_offset_y),
	    coverage_flags, pair(hw.GetColorControl().op, blend.enable),
	    pair(masks.masks[0], masks.masks[1]), pair(masks.shader_sample_exclusion, masks.observed),
	    pair(z.stencil_info.htile_stencil_disabled,
	         static_cast<uint32_t>(z.stencil_info.texture_compatibility)),
	});
	using Key = decltype(key);
	// Surface addresses rotate; retain distinct format, shader, and draw-coverage states.
	static std::mutex mutex;
	static std::array<Key, 64> configurations {};
	static uint32_t count = 0;
	static bool limit_logged = false;
	const std::lock_guard lock(mutex);
	const auto end = configurations.begin() + count;
	if (std::find(configurations.begin(), end, key) != end) {
		return;
	}
	if (count == configurations.size()) {
		if (!limit_logged) {
			LOGF("DepthCopyTrace: distinct-configuration limit reached (64)\n");
			limit_logged = true;
		}
		return;
	}
	const auto id = count++;
	configurations[id] = key;
	LOGF("DepthCopyTrace[%u]: frame=%d ps=0x%016" PRIx64
	     " depth=%u stencil=%u centroid=%u sample=%u clear=%u/%u decompress=%u/%u"
	     " resummarize=%u ztest=%u zwrite=%u zfunc=%u stest=%u bounds=%u write_disable=%u/%u\n",
	     id, buffer.GetContext().GetGpu().GetFrameNum(), ps,
	     static_cast<unsigned>(rc.copy_depth_to_color), static_cast<unsigned>(rc.copy_stencil_to_color),
	     static_cast<unsigned>(rc.copy_centroid), rc.copy_sample,
	     static_cast<unsigned>(rc.depth_clear_enable), static_cast<unsigned>(rc.stencil_clear_enable),
	     static_cast<unsigned>(rc.depth_compress_disable), static_cast<unsigned>(rc.stencil_compress_disable),
	     static_cast<unsigned>(rc.resummarize_enable), static_cast<unsigned>(dc.z_enable),
	     static_cast<unsigned>(dc.z_write_enable), dc.zfunc, static_cast<unsigned>(dc.stencil_enable),
	     static_cast<unsigned>(dc.depth_bounds_enable), static_cast<unsigned>(z.depth_view.depth_write_disable),
	     static_cast<unsigned>(z.depth_view.stencil_write_disable));
	LOGF("DepthCopyTrace[%u]: source z=0x%016" PRIx64 "/0x%016" PRIx64
	     " stencil=0x%016" PRIx64 "/0x%016" PRIx64
	     " formats=%u/%u extent=%ux%u valid=%u samples=%u(enc=%u) mip=%u layers=%u..%u"
	     " htile=0x%016" PRIx64 " enabled=%u stencil_disabled=%u stencil_compressed=%u\n",
	     id, z.z_read_base_addr, z.z_write_base_addr, z.stencil_read_base_addr, z.stencil_write_base_addr,
	     static_cast<unsigned>(z.z_info.format), static_cast<unsigned>(z.stencil_info.format),
	     z.size.x_max + 1u, z.size.y_max + 1u, static_cast<unsigned>(z.size.valid),
	     render_sample_count(z.z_info.num_samples), z.z_info.num_samples,
	     z.depth_view.current_mip_level, z.depth_view.slice_start, z.depth_view.slice_max, z.htile_data_base_addr,
	     static_cast<unsigned>(z.z_info.htile_acceleration),
	     static_cast<unsigned>(z.stencil_info.htile_stencil_disabled),
	     static_cast<unsigned>(z.stencil_info.format != Prospero::StencilFormat::kInvalid &&
	                           z.z_info.htile_acceleration && !z.stencil_info.htile_stencil_disabled));
	LOGF("DepthCopyTrace[%u]: cb0=0x%016" PRIx64
	     " format=%u nfmt=%u order=%u tile=%u extent=%ux%ux%u dimension=%u samples_enc=%u fragments_enc=%u"
	     " mip=%u layers=%u..%u target_mask=0x%x shader_mask=0x%x output_mode=%u color_mode=%u alias_z=%u alias_stencil=%u\n",
	     id, cb.base.addr, static_cast<unsigned>(cb.info.format), static_cast<unsigned>(cb.info.channel_type),
	     static_cast<unsigned>(cb.info.channel_order), static_cast<unsigned>(cb.attrib3.tile_mode),
	     cb.attrib2.width + 1u, cb.attrib2.height + 1u, cb.attrib3.depth + 1u, cb.attrib3.dimension,
	     cb.attrib.num_samples, cb.attrib.num_fragments, cb.view.current_mip_level,
	     cb.view.base_array_slice_index, cb.view.last_array_slice_index,
	     hw.GetRenderTargetMask() & 0xfu, sh.m_cbShaderMask & 0xfu, sh.target_output_mode[0], hw.GetColorControl().mode,
	     static_cast<unsigned>(z.z_read_base_addr == cb.base.addr),
	     static_cast<unsigned>(z.stencil_read_base_addr == cb.base.addr));
	LOGF("DepthCopyTrace[%u]: es=0x%016" PRIx64 " gs=0x%016" PRIx64
	     " prim=%u index_offset=%u submit=%" PRIu64 "\n",
	     id, vs.es_regs.data_addr, vs.gs_regs.data_addr, static_cast<unsigned>(ucfg.GetPrimType()),
	     ucfg.GetIndexOffset(), debug.submit_id);
	// At ResolveRenderDepthTarget, DrawAuto/DrawIndex have recorded their arguments but have
	// not entered ExecutePreparedDraw (which reuses these fields for phase diagnostics).
	// Indirect draws call the same entry points; their origin is not retained in this snapshot.
	if (debug.op == static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto) && debug.args[1] == 0) {
		LOGF("DepthCopyTrace[%u]: draw=auto origin=unrecorded count=%u instances=%u"
		     " first_vertex=%u first_instance=%" PRIu64 "\n",
		     id, debug.args[0], debug.args[3], debug.args[2], debug.arg4);
	} else if (debug.op == static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex) &&
	           debug.args[1] == 0 && debug.args[2] == 1) {
		LOGF("DepthCopyTrace[%u]: draw=indexed origin=unrecorded count=%u instances=%u"
		     " index_addr=0x%016" PRIx64 " index_type/base_vertex/first_instance=unrecorded\n",
		     id, debug.args[0], debug.args[3], debug.arg4);
	} else {
		LOGF("DepthCopyTrace[%u]: draw=unknown op=%u args=%u/%u/%u/%u/0x%016" PRIx64 "\n",
		     id, debug.op, debug.args[0], debug.args[1], debug.args[2], debug.args[3], debug.arg4);
	}
	LOGF("DepthCopyTrace[%u]: viewport0_scale=(%.9g,%.9g,%.9g) offset=(%.9g,%.9g,%.9g)"
	     " transform=0x%x scissor0=(%d,%d)-(%d,%d) clip_rule=0x%x"
	     " window_offset=%d/%d hw_offset=%u/%u\n",
	     id, vp0.xscale, vp0.yscale, vp0.zscale, vp0.xoffset, vp0.yoffset, vp0.zoffset,
	     vp.transform_control, scissor.left, scissor.top, scissor.right, scissor.bottom,
	     vp.clip_rect_rule, vp.window_offset_x, vp.window_offset_y, vp.hw_offset_x, vp.hw_offset_y);
	LOGF("DepthCopyTrace[%u]: cull=%u/%u face=%u polygon=%u/%u/%u vertex_window_offset=%u"
	     " clip_disable=%u dx_clip=%u z_clip_disable=%u/%u force_vs_viewport=%u"
	     " user_clip=%u/%u msaa=%u viewport_scissor=%u color_op=0x%x blend=%u"
	     " aa_masks=0x%08x/0x%08x ps_sample_exclusion=0x%08x masks_observed=0x%x\n",
	     id, static_cast<unsigned>(mode.cull_front), static_cast<unsigned>(mode.cull_back),
	     static_cast<unsigned>(mode.face), mode.poly_mode, mode.polymode_front_ptype,
	     mode.polymode_back_ptype, static_cast<unsigned>(mode.vtx_window_offset_enable),
	     static_cast<unsigned>(clip.clip_disable), static_cast<unsigned>(clip.dx_clip_space),
	     static_cast<unsigned>(clip.min_z_clip_disable), static_cast<unsigned>(clip.max_z_clip_disable),
	     static_cast<unsigned>(clip.force_viewport_index_from_vs_enable), clip.user_clip_planes,
	     clip.user_clip_plane_mode, static_cast<unsigned>(scan.msaa_enable),
	     static_cast<unsigned>(scan.vport_scissor_enable), hw.GetColorControl().op,
	     static_cast<unsigned>(blend.enable), masks.masks[0], masks.masks[1],
	     masks.shader_sample_exclusion, masks.observed);
}

[[noreturn]] static void DepthFatal(const char* format, ...) {
	std::fputs("Depth target fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	EXIT("unsupported render state; details were printed above\n");
}

static vk::StencilOp ConvertStencilOp(uint8_t value, uint8_t write_mask, uint8_t op_value) {
	if (write_mask == 0) {
		return vk::StencilOp::eKeep;
	}
	switch (static_cast<Prospero::StencilOp>(value)) {
		case Prospero::StencilOp::kKeep: return vk::StencilOp::eKeep;
		case Prospero::StencilOp::kZero: return vk::StencilOp::eZero;
		case Prospero::StencilOp::kReplaceTest: return vk::StencilOp::eReplace;
		case Prospero::StencilOp::kReplaceOp:
			if ((op_value & write_mask) == 0) {
				return vk::StencilOp::eZero;
			}
			return vk::StencilOp::eReplace;
		case Prospero::StencilOp::kAddClamp: return vk::StencilOp::eIncrementAndClamp;
		case Prospero::StencilOp::kSubClamp: return vk::StencilOp::eDecrementAndClamp;
		case Prospero::StencilOp::kInvert: return vk::StencilOp::eInvert;
		case Prospero::StencilOp::kAddWrap: return vk::StencilOp::eIncrementAndWrap;
		case Prospero::StencilOp::kSubWrap: return vk::StencilOp::eDecrementAndWrap;
		case Prospero::StencilOp::kXor:
			if ((write_mask & op_value) == 0) {
				return vk::StencilOp::eKeep;
			}
			if ((write_mask & ~op_value) != 0) {
				DepthFatal("unsupported stencil XOR operands: write mask=0x%02" PRIx8
				           ", operation value=0x%02" PRIx8,
				           write_mask, op_value);
			}
			return vk::StencilOp::eInvert;
		default: DepthFatal("unsupported stencil operation: 0x%02" PRIx8, value);
	}
}

static vk::StencilOpState ConvertStencilState(
    uint8_t compare, const std::array<uint8_t, 3>& operations, uint8_t op_value,
    const vk::StencilOpState& state) {
	const auto test_value = state.reference;
	auto reference       = test_value;
	auto required_bits   = state.compareMask;
	if (compare == static_cast<uint8_t>(vk::CompareOp::eAlways) ||
	    compare == static_cast<uint8_t>(vk::CompareOp::eNever)) {
		required_bits = 0;
	}
	std::array<vk::StencilOp, 3> converted {};
	for (size_t i = 0; i < operations.size(); i++) {
		converted[i] = ConvertStencilOp(operations[i], state.writeMask, op_value);
		if (converted[i] != vk::StencilOp::eReplace) {
			continue;
		}
		auto replacement = test_value;
		if (static_cast<Prospero::StencilOp>(operations[i]) == Prospero::StencilOp::kReplaceOp) {
			replacement = op_value;
		}
		if (((reference ^ replacement) & required_bits & state.writeMask) != 0) {
			DepthFatal("unsupported stencil replacement: compare=%u, compare mask=0x%02" PRIx32
			           ", write mask=0x%02" PRIx32 ", operation value=0x%02" PRIx8
			           ", test value=0x%02" PRIx32,
			           compare, state.compareMask, state.writeMask, op_value, test_value);
		}
		// Vulkan shares one reference between comparison and every replacement on this face.
		reference = (reference & ~state.writeMask) | (replacement & state.writeMask);
		required_bits |= state.writeMask;
	}
	return {converted[0], converted[1], converted[2], static_cast<vk::CompareOp>(compare),
	        state.compareMask, state.writeMask, reference};
}

[[nodiscard]] static vk::Format ResolveHostDepthAttachmentFormat(const CommandBuffer&     buffer,
                                                                 const DepthFormatPolicy& policy,
                                                                 bool     has_stencil,
                                                                 uint32_t samples) {
	auto&      graphics         = buffer.GetGraphics();
	const auto required_samples = vulkan_sample_count(samples);
	const auto supports         = [&](vk::Format format) {
		vk::ImageFormatProperties properties {};
		return format != vk::Format::eUndefined &&
		       graphics.GetImageFormatProperties(
		           format, vk::ImageType::e2D, vk::ImageTiling::eOptimal, DepthTargetImageUsage(),
		           vk::ImageCreateFlags {}, &properties) == vk::Result::eSuccess &&
		       static_cast<bool>(properties.sampleCounts & required_samples);
	};
	if (!has_stencil) {
		return supports(policy.depth_attachment_format) ? policy.depth_attachment_format
		                                                : vk::Format::eUndefined;
	}
	for (const auto format: policy.stencil_attachment_formats) {
		if (supports(format)) {
			return format;
		}
	}
	return vk::Format::eUndefined;
}

static TextureCache::ImageDesc MakeDepthTargetDesc(const CommandBuffer& buffer,
                                                  const HW::DepthRenderTarget& z,
                                                  bool write_buffer = false) {
	const bool has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	const auto depth_address = write_buffer ? z.z_write_base_addr : z.z_read_base_addr;
	const auto stencil_address =
	    write_buffer ? z.stencil_write_base_addr : z.stencil_read_base_addr;
	if (!z.z_info.HasValidTextureCompatibility() ||
	    !z.stencil_info.HasValidTextureCompatibility()) {
		DepthFatal("invalid PS5 depth texture-compatibility encoding");
	}
	const bool has_htile = z.z_info.htile_acceleration;
	const bool unsupported_shading_rate_encoding =
	    z.shading_rate_encoding > 1 || (z.shading_rate_encoding != 0 && !has_htile);
	const auto samples   = render_sample_count(z.z_info.num_samples);
	if (samples == 0) {
		DepthFatal("unsupported depth fragment count: %u", z.z_info.num_samples);
	}
	const bool htile_stencil_compat = depth_htile_stencil_acceleration_compatible(
	    has_stencil, has_htile, z.stencil_info.htile_stencil_disabled);
	const auto view = ResolveTargetViewInfo(z.depth_view.slice_start, z.depth_view.slice_max);
	switch (view.type) {
		case TargetViewType::Image2D:
		case TargetViewType::Image2DArray: break;
		case TargetViewType::Unsupported:
			DepthFatal("invalid depth view: base=%u last=%u", z.depth_view.slice_start,
			           z.depth_view.slice_max);
	}
	// EXPCLEAR permits an HTile acceleration state; the host attachment is already expanded.
	if (z.z_info.max_mip_level != 0 ||
	    z.depth_view.current_mip_level != 0 || unsupported_shading_rate_encoding ||
	    depth_address == 0 || (depth_address & 0xffffu) != 0) {
		DepthFatal("unsupported depth register state");
	}
	if (has_stencil) {
		if (z.stencil_info.format != Prospero::StencilFormat::k8UInt || !htile_stencil_compat ||
		    stencil_address == 0 || (stencil_address & 0xffffu) != 0) {
			DepthFatal("unsupported stencil attachment state");
		}
	} else if (z.stencil_read_base_addr != 0 || z.stencil_write_base_addr != 0) {
		DepthFatal("stencil state without an active stencil attachment");
	}
	if (has_htile) {
		if (z.htile_data_base_addr == 0 || (z.htile_data_base_addr & 0x7fffu) != 0) {
			DepthFatal("invalid HTile metadata address");
		}
		if (z.depth_view.slice_max >= 32) {
			DepthFatal("HTile clear tracking supports at most 32 slices");
		}
	}
	if (!z.size.valid) {
		DepthFatal("missing depth extent");
	}
	const uint32_t width  = static_cast<uint32_t>(z.size.x_max) + 1u;
	const uint32_t height = static_cast<uint32_t>(z.size.y_max) + 1u;
	if (width > 16384 || height > 16384) {
		DepthFatal("invalid depth extent");
	}
	const auto* policy = FindDepthFormatPolicy(z.z_info.format);
	if (policy == nullptr) {
		DepthFatal("unsupported depth/stencil format pair");
	}
	const auto ideal_format = DepthAttachmentFormat(*policy, has_stencil);
	const auto format = ResolveHostDepthAttachmentFormat(buffer, *policy, has_stencil, samples);
	if (format == vk::Format::eUndefined) {
		DepthFatal("no host depth/stencil format supports required usage for %s",
		           vk::to_string(ideal_format).c_str());
	}
	const auto     guest_format = policy->guest_format;
	const uint32_t bytes        = policy->bytes_per_element;
	const auto     pitch        = TileGetDepthPitch(width, bytes, z.z_info.num_samples);
	TileSizeAlign depth_size {};
	TileSizeAlign stencil_size {};
	TileSizeAlign htile_size {};
	if (!TileGetDepthSize(width, height, 0, z.z_info.format, z.stencil_info.format, has_htile,
	                      stencil_size, htile_size, depth_size, z.z_info.num_samples) ||
	    depth_size.align != 65536 || depth_size.size == 0 ||
	    (has_stencil != (stencil_size.align == 65536 && stencil_size.size != 0)) ||
	    (has_htile != (htile_size.align == 32768 && htile_size.size != 0))) {
		DepthFatal("unsupported depth/stencil/HTile footprint");
	}
	if (depth_size.size > UINT64_MAX / view.image_layers ||
	    stencil_size.size > UINT64_MAX / view.image_layers ||
	    htile_size.size > UINT64_MAX / view.image_layers) {
		DepthFatal("layered depth footprint overflow");
	}
	const auto depth_backing_size   = depth_size.size * view.image_layers;
	const auto stencil_backing_size = stencil_size.size * view.image_layers;
	const auto htile_backing_size   = htile_size.size * view.image_layers;
	if (!GuestRange {depth_address, depth_backing_size}.Valid() ||
	    (has_stencil && !GuestRange {stencil_address, stencil_backing_size}.Valid()) ||
	    (has_htile && !GuestRange {z.htile_data_base_addr, htile_backing_size}.Valid())) {
		DepthFatal("layered depth backing range is invalid");
	}
	TextureCache::ImageDesc desc {};
	desc.type                 = TextureCache::BindingType::DepthTarget;
	desc.info.data            = {depth_address, depth_backing_size};
	desc.info.stencil =
	    has_stencil ? GuestRange {stencil_address, stencil_backing_size} : GuestRange {};
	desc.info.pixel_format    = format;
	desc.info.guest_format    = guest_format;
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {width, height, 1};
	desc.info.resources       = {1, view.image_layers};
	desc.info.pitch           = pitch;
	desc.info.bytes_per_block = bytes;
	desc.info.samples         = samples;
	desc.info.tile_mode       = Prospero::TileMode::kDepth;
	desc.info.UpdateSize();
	desc.info.metadata.range =
	    has_htile ? GuestRange {z.htile_data_base_addr, htile_backing_size} : GuestRange {};
	desc.info.metadata.kind   = has_htile ? ImageMetadataKind::Htile : ImageMetadataKind::None;
	desc.info.metadata.stencil_compressed =
	    has_stencil && has_htile && !z.stencil_info.htile_stencil_disabled;
	desc.view_info.format = format;
	desc.view_info.type =
	    view.layer_count == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	desc.view_info.aspect      = ImageViewOps::DepthAspectMask(format);
	desc.view_info.base_level  = 0;
	desc.view_info.level_count = 1;
	desc.view_info.base_layer  = view.base_layer;
	desc.view_info.layer_count = view.layer_count;
	desc.view_info.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
	return desc;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& r) {
	KYTY_PROFILER_FUNCTION();
	const auto& hw          = buffer.GetRegisters();
	const auto& z           = hw.GetDepthRenderTarget();
	const auto& rc          = hw.GetRenderControl();
	const auto& dc          = hw.GetDepthControl();
	const auto& sc          = hw.GetStencilControl();
	const auto& sm          = hw.GetStencilMask();
	const bool  has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	const bool depth_active = dc.z_enable || dc.depth_bounds_enable ||
	                          rc.depth_clear_enable || rc.copy_depth_to_color;
	const bool stencil_active =
	    has_stencil && (dc.stencil_enable || rc.stencil_clear_enable || rc.copy_stencil_to_color);
	if (!depth_active && !stencil_active) {
		return;
	}
	const bool attachment_unbound =
	    z.z_info.format == Prospero::DepthFormat::kInvalid &&
	    z.stencil_info.format == Prospero::StencilFormat::kInvalid && z.z_info.num_samples == 0 &&
	    z.z_info.texture_compatibility == Prospero::TextureCompatiblePlaneCompression::kDisable &&
	    !z.z_info.expclear_enabled && !z.z_info.partially_resident && z.z_info.max_mip_level == 0 &&
	    z.stencil_info.texture_compatibility == Prospero::TextureCompatibleStencil::kDisable &&
	    !z.stencil_info.expclear_enabled && !z.stencil_info.partially_resident &&
	    z.depth_view.slice_start == 0 && z.depth_view.slice_max == 0 &&
	    z.depth_view.current_mip_level == 0 && !z.depth_view.depth_write_disable &&
	    !z.depth_view.stencil_write_disable && z.z_read_base_addr == 0 && z.z_write_base_addr == 0 &&
	    z.stencil_read_base_addr == 0 && z.stencil_write_base_addr == 0 &&
	    z.htile_data_base_addr == 0 &&
	    // DB_DEPTH_SIZE_XY is independent state and may remain programmed after the attachment
	    // formats and addresses are unbound. A zero encoding is the valid 1x1 value, so its
	    // presence alone must not manufacture a depth attachment.
	    !z.z_info.htile_acceleration && z.shading_rate_encoding == 0 && z.size.x_max == 0 &&
	    z.size.y_max == 0;
	if (attachment_unbound) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("DepthTarget: ignoring enabled depth/stencil state without a bound attachment\n");
		}
		return;
	}
    // Boot-only fallback for PS VR2 games: these copy-to-color modes need a
    // dedicated Vulkan path. Preserve the depth attachment for initialization.
    // The result can have incorrect depth/stencil-derived colors.
    if (rc.copy_depth_to_color || rc.copy_stencil_to_color) {
        if (!Config::VrEnabled()) {
            DepthFatal("unsupported copy-to-color mode");
        }
        TraceDepthCopyState(buffer);
        static std::atomic_bool logged = false;
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            LOGF("DepthTarget: VR copy-to-color mode unsupported, continuing with ordinary attachment\n");
        }
    }
    if (rc.copy_sample != 0 || dc.zfunc > static_cast<uint8_t>(vk::CompareOp::eAlways) ||
	    (!z.depth_view.depth_write_disable && z.z_write_base_addr != z.z_read_base_addr) ||
	    (has_stencil && !z.depth_view.stencil_write_disable &&
	     z.stencil_write_base_addr != z.stencil_read_base_addr)) {
		DepthFatal("unsupported depth register state: copy_depth=%d copy_stencil=%d copy_centroid=%d copy_sample=%u zfunc=%u depth_diff=%d stencil_diff=%d",
                       static_cast<int>(rc.copy_depth_to_color),
                       static_cast<int>(rc.copy_stencil_to_color),
                       static_cast<int>(rc.copy_centroid),
                       static_cast<unsigned>(rc.copy_sample),
                       static_cast<unsigned>(dc.zfunc),
                       static_cast<int>(!z.depth_view.depth_write_disable && z.z_write_base_addr != z.z_read_base_addr),
                       static_cast<int>(has_stencil && !z.depth_view.stencil_write_disable &&
                                        z.stencil_write_base_addr != z.stencil_read_base_addr));
	}
	r.desc = MakeDepthTargetDesc(buffer, z);
	r.depth_clear_enable      = rc.depth_clear_enable;
	r.depth_load_clear_enable = r.depth_clear_enable;
	r.depth_clear_value       = hw.GetDepthClearValue();
	r.depth_test_enable       = dc.z_enable;
	r.depth_write_enable      = r.depth_test_enable && dc.z_write_enable &&
	                            !z.depth_view.depth_write_disable && !r.depth_clear_enable;
	r.depth_compare_op        = static_cast<vk::CompareOp>(dc.zfunc);

	r.depth_bounds_test_enable = dc.depth_bounds_enable;

	r.stencil_clear_enable =
	    has_stencil && rc.stencil_clear_enable && !z.depth_view.stencil_write_disable;
	r.stencil_clear_value = hw.GetStencilClearValue();
	r.stencil_test_enable = has_stencil && dc.stencil_enable;
	if (r.stencil_test_enable) {
		const bool stencil_ops_disabled =
		    rc.stencil_clear_enable || z.depth_view.stencil_write_disable;
		const uint8_t front_write_mask = stencil_ops_disabled ? 0 : sm.stencil_writemask;
		const uint8_t back_write_mask  = stencil_ops_disabled ? 0 : sm.stencil_writemask_bf;
		if (dc.stencilfunc > static_cast<uint8_t>(vk::CompareOp::eAlways) ||
		    (dc.backface_enable &&
		     dc.stencilfunc_bf > static_cast<uint8_t>(vk::CompareOp::eAlways))) {
			DepthFatal("unsupported stencil compare state");
		}
		r.stencil_front = ConvertStencilState(
		    dc.stencilfunc, {sc.stencil_fail, sc.stencil_zpass, sc.stencil_zfail},
		    sm.stencil_opval, {.compareMask = sm.stencil_mask, .writeMask = front_write_mask,
		                       .reference = sm.stencil_testval});
		if (dc.backface_enable) {
			r.stencil_back = ConvertStencilState(
			    dc.stencilfunc_bf, {sc.stencil_fail_bf, sc.stencil_zpass_bf, sc.stencil_zfail_bf},
			    sm.stencil_opval_bf, {.compareMask = sm.stencil_mask_bf, .writeMask = back_write_mask,
			                          .reference = sm.stencil_testval_bf});
		} else {
			r.stencil_back = r.stencil_front;
		}
	}
	auto& cache = m_context.GetTextureCache();
	r.image_id = cache.FindImage(r.desc);
	BindRenderTarget(r.image_id);
}

namespace {

struct DepthStencilColorCopyPlan {
	vk::ImageAspectFlagBits aspect;
	vk::Format color_format;
	vk::Rect2D region;
	ImageInfo color_info;
};

enum class DepthCopyFallback : uint32_t {
	Draw, Source, Color, Raster, EmptyScissor, VertexShader, TargetMapping, PixelShader,
	TargetTiling, AliasedRanges, Discovery, NativeBacking,
};

void TraceDepthCopyFallback(const CommandBuffer& buffer, DepthCopyFallback reason) {
	if (!DepthCopyTraceEnabled()) {
		return;
	}
	const auto& hw = buffer.GetRegisters();
	const auto& rc = hw.GetRenderControl();
	const auto& sh = hw.GetShaderRegisters();
	const auto& db = sh.db_shader_control;
	const auto& vs = buffer.GetShaders().GetVs();
	const uint64_t bit = uint64_t {1} << (static_cast<uint32_t>(reason) * 2u +
	                                    static_cast<uint32_t>(rc.copy_stencil_to_color));
	static std::atomic<uint64_t> logged {0};
	if ((logged.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) {
		return;
	}
	static constexpr const char* names[] = {
	    "draw-or-copy-state", "source-plane-state", "color-or-export-state", "raster-coverage",
	    "empty-scissor", "vertex-template-or-inputs", "target-channel-mapping", "pixel-template",
	    "target-tile-alignment", "overlapping-guest-ranges", "image-discovery", "native-copy-state",
	};
	uint32_t output_modes = 0;
	for (uint32_t i = 0; i < std::size(sh.target_output_mode); i++) {
		output_modes |= static_cast<uint32_t>(sh.target_output_mode[i]) << (i * 4u);
	}
	LOGF("DepthCopy: fallback reason=%s depth=%u stencil=%u stages=0x%x lds=%u vs_out=0x%x"
	     " db_kill/z/mask/dual/noop=%u/%u/%u/%u/%u aa=%u/%u"
	     " target_mask=0x%08x shader_mask=0x%08x output_modes=0x%08x\n",
	     names[static_cast<uint32_t>(reason)], static_cast<unsigned>(rc.copy_depth_to_color),
	     static_cast<unsigned>(rc.copy_stencil_to_color), hw.GetShaderStages(), vs.gs_regs.rsrc2.lds_size,
	     sh.m_paClVsOutCntl, static_cast<unsigned>(db.shader_kill_enable),
	     static_cast<unsigned>(db.shader_z_export_enable), static_cast<unsigned>(db.shader_mask_export_enable),
	     static_cast<unsigned>(db.shader_dual_export_enable), static_cast<unsigned>(db.shader_execute_on_noop),
	     hw.GetAaConfig().msaa_num_samples, hw.GetAaConfig().msaa_exposed_samples,
	     hw.GetRenderTargetMask(), sh.m_cbShaderMask, output_modes);
}

// This template emits (-1,-1), (1,-1), (-1,1), z=0, w=1 from VertexIndex.
// RectList assembly supplies the fourth corner. Match all executed instructions,
// including the NGG bookkeeping prefix, rather than a title or guest address.
constexpr std::array<uint32_t, 27> depth_copy_vertex_code {
	0xbfa00001u, 0x93ebff03u, 0x00080008u, 0x8700ff03u, 0x000000ffu, 0x8f6a8c6bu,
	0x887c6a00u, 0xbf900009u, 0x81ea6bc0u, 0x90fe6ac1u, 0xf8000941u, 0x00000000u,
	0x81ea00c0u, 0xbf8cff0fu, 0x90fe6ac1u, 0x34040a81u, 0x36060ac2u, 0x7e000280u,
	0x7e0202f2u, 0x36040482u, 0x4a0606c1u, 0x4a0404c1u, 0x7e060b03u, 0x7e040b02u,
	0xf80008cfu, 0x01000302u, 0xbf810000u,
};
// The export only triggers the DB copy; its zero values are replaced by the selected plane.
constexpr std::array<uint32_t, 4> depth_copy_pixel_code {
	0x7e000280u, 0xf8001803u, 0x00000000u, 0xbf810000u,
};

std::optional<DepthStencilColorCopyPlan> PlanDepthStencilColorCopy(
    const CommandBuffer& buffer, const DrawAutoArgs& args, DepthCopyFallback& reason) {
	const auto reject = [&](DepthCopyFallback why) -> std::optional<DepthStencilColorCopyPlan> {
		reason = why;
		return std::nullopt;
	};
	const auto& hw = buffer.GetRegisters();
	const auto& rc = hw.GetRenderControl();
	const auto& dc = hw.GetDepthControl();
	const auto& z = hw.GetDepthRenderTarget();
	const auto& cb = hw.GetRenderTarget(0);
	const auto& sh = hw.GetShaderRegisters();
	const auto& vs = buffer.GetShaders().GetVs();
	const auto& ps = buffer.GetShaders().GetPs();
	const auto& ucfg = buffer.GetUserConfig();
	if (rc.copy_depth_to_color == rc.copy_stencil_to_color || !rc.copy_centroid ||
	    rc.copy_sample != 0 || rc.depth_clear_enable || rc.stencil_clear_enable ||
	    rc.depth_compress_disable || rc.stencil_compress_disable || rc.resummarize_enable ||
	    dc.z_enable || dc.z_write_enable || dc.stencil_enable || dc.depth_bounds_enable ||
	    args.vertex_count != 3 || args.instance_count != 1 || args.first_vertex != 0 ||
	    args.first_instance != 0 || args.render_target_slice_offset != 0 ||
	    ucfg.GetPrimType() != Prospero::PrimitiveType::kRectList || ucfg.GetIndexOffset() != 0 ||
	    (hw.GetShaderStages() & 0x20u) != 0 || vs.gs_regs.rsrc2.lds_size != 0 ||
	    vs.es_regs.data_addr == 0 || ps.ps_regs.data_addr == 0) {
		return reject(DepthCopyFallback::Draw);
	}
	const bool stencil = rc.copy_stencil_to_color;
	const bool has_stencil = z.stencil_info.format == Prospero::StencilFormat::k8UInt;
	if (z.z_info.format != Prospero::DepthFormat::kZ32F || z.z_info.num_samples != 0 ||
	    !z.size.valid || z.size.x_max >= 16384 || z.size.y_max >= 16384 ||
	    !z.z_info.HasValidTextureCompatibility() || !z.stencil_info.HasValidTextureCompatibility() ||
	    z.z_info.partially_resident || z.stencil_info.partially_resident ||
	    z.z_info.max_mip_level != 0 || z.depth_view.current_mip_level != 0 ||
	    z.depth_view.slice_start != 0 || z.depth_view.slice_max != 0 ||
	    z.depth_view.depth_write_disable || z.depth_view.stencil_write_disable ||
	    z.shading_rate_encoding != 0 || z.z_read_base_addr == 0 ||
	    (z.z_read_base_addr & 0xffffu) != 0 || z.z_read_base_addr != z.z_write_base_addr ||
	    (has_stencil ? (z.stencil_read_base_addr == 0 ||
	                    (z.stencil_read_base_addr & 0xffffu) != 0 ||
	                    z.stencil_read_base_addr != z.stencil_write_base_addr)
	                 : (z.stencil_info.format != Prospero::StencilFormat::kInvalid ||
	                    z.stencil_read_base_addr != 0 || z.stencil_write_base_addr != 0)) ||
	    (z.z_info.htile_acceleration &&
	     (z.htile_data_base_addr == 0 || (z.htile_data_base_addr & 0x7fffu) != 0)) ||
	    !depth_htile_stencil_acceleration_compatible(
	        has_stencil, z.z_info.htile_acceleration, z.stencil_info.htile_stencil_disabled) ||
	    (stencil && (!has_stencil ||
	                 (z.z_info.htile_acceleration && !z.stencil_info.htile_stencil_disabled)))) {
		return reject(DepthCopyFallback::Source);
	}
	const auto& db = sh.db_shader_control;
	const uint32_t logical_mask = stencil ? 2u : 1u;
	if (hw.GetColorControl().mode != 1 || hw.GetColorControl().op != 0xcc ||
	    hw.GetBlendControl(0).enable || hw.GetRenderTargetMask() != logical_mask ||
	    sh.m_cbShaderMask != 3 || sh.target_output_mode[0] != 2 ||
	    !std::all_of(std::begin(sh.target_output_mode) + 1, std::end(sh.target_output_mode),
	                 [](uint8_t mode) { return mode == 0; }) ||
	    sh.shader_z_format != 0 || sh.m_paClVsOutCntl != 0 ||
	    db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	    db.shader_dual_export_enable || db.shader_execute_on_noop ||
	    cb.base.addr == 0 || cb.attrib.num_samples != 0 || cb.attrib.num_fragments != 0 ||
	    cb.attrib3.dimension != 1 || cb.attrib3.depth != 0 || cb.attrib2.num_mip_levels != 0 ||
	    cb.view.current_mip_level != 0 || cb.view.base_array_slice_index != 0 ||
	    cb.view.last_array_slice_index != 0 || cb.attrib2.width != z.size.x_max ||
	    cb.attrib2.height != z.size.y_max || cb.info.fmask_compression_enable ||
	    cb.info.fmask_data_compression_disable || cb.info.fmask_one_frag_mode ||
	    cb.info.cmask_fast_clear_enable || cb.info.dcc_compression_enable ||
	    cb.cmask.addr != 0 || cb.fmask.addr != 0 || cb.dcc_addr.addr != 0 ||
	    cb.dcc.data_write_on_dcc_clear_to_reg ||
	    (stencil ? (cb.info.format != Prospero::ChannelLayout::k8 ||
	                cb.info.channel_type != Prospero::ChannelType::kUInt ||
	                cb.info.channel_order != Prospero::ChannelOrder::kAlt)
	             : (cb.info.format != Prospero::ChannelLayout::k32 ||
	                cb.info.channel_type != Prospero::ChannelType::kFloat ||
	                cb.info.channel_order != Prospero::ChannelOrder::kStandard))) {
		return reject(DepthCopyFallback::Color);
	}
	switch (cb.attrib3.tile_mode) {
		case Prospero::TileMode::kLinear:
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
		case Prospero::TileMode::kDepth:
		case Prospero::TileMode::kRenderTarget: break;
		default: return reject(DepthCopyFallback::TargetTiling);
	}
	const auto& mode = hw.GetModeControl();
	const auto& clip = hw.GetClipControl();
	const auto& scan = hw.GetScanModeControl();
	const auto& aa = hw.GetAaMaskControl();
	const auto& vp = hw.GetScreenViewport();
	const auto& vp0 = vp.viewports[0];
	const uint32_t width = cb.attrib2.width + 1u;
	const uint32_t height = cb.attrib2.height + 1u;
	if (mode.cull_front || mode.cull_back || mode.poly_mode != 0 ||
	    mode.polymode_front_ptype != 0 || mode.polymode_back_ptype != 0 ||
	    mode.vtx_window_offset_enable || clip.clip_disable || clip.dx_clip_space ||
	    clip.vertex_kill_any || clip.user_clip_planes != 0 || clip.user_clip_plane_mode != 0 ||
	    clip.force_viewport_index_from_vs_enable || clip.min_z_clip_disable || clip.max_z_clip_disable ||
	    scan.msaa_enable || scan.line_stipple_enable || hw.GetAaConfig().msaa_num_samples != 0 ||
	    hw.GetAaConfig().msaa_exposed_samples != 0 || aa.masks[0] != UINT32_MAX ||
	    aa.masks[1] != UINT32_MAX || aa.shader_sample_exclusion != 0 ||
	    vp.transform_control != 0x43fu || vp.clip_rect_rule != 0xffffu ||
	    vp0.xscale != static_cast<float>(width) * 0.5f || vp0.xoffset != vp0.xscale ||
	    vp0.yscale != -static_cast<float>(height) * 0.5f || vp0.yoffset != -vp0.yscale ||
	    vp0.zscale != 0.5f || vp0.zoffset != 0.5f ||
	    !std::isfinite(hw.GetDepthClearValue()) || hw.GetDepthClearValue() < 0.0f ||
	    hw.GetDepthClearValue() > 1.0f) {
		return reject(DepthCopyFallback::Raster);
	}
	const auto scissor = calc_final_scissor(vp, scan, {width, height}, 0);
	if (scissor.right <= scissor.left || scissor.bottom <= scissor.top) {
		return reject(DepthCopyFallback::EmptyScissor);
	}
	ShaderVertexInputInfo vertex_info {};
	const auto vertex = PrepareProgram(vs, hw, ucfg, vertex_info);
	if (vertex_info.logical_stage != ShaderType::Vertex || vertex_info.fetch_embedded ||
	    vertex_info.fetch_external || vertex_info.resources_num != 0 ||
	    vertex.code.size() < depth_copy_vertex_code.size() ||
	    !std::equal(depth_copy_vertex_code.begin(), depth_copy_vertex_code.end(), vertex.code.begin())) {
		return reject(DepthCopyFallback::VertexShader);
	}
	const auto format = TextureGetRenderTargetFormat(cb.info.format, cb.info.channel_type,
	                                                cb.info.channel_order);
	if (format.export_mapping.ApplyMask(logical_mask & sh.m_cbShaderMask) != 1u) {
		return reject(DepthCopyFallback::TargetMapping);
	}
	if (cb.attrib3.tile_mode == Prospero::TileMode::kStandard4KB ||
	    cb.attrib3.tile_mode == Prospero::TileMode::kStandard64KB ||
	    cb.attrib3.tile_mode == Prospero::TileMode::kDepth) {
		TileTextureBlockLayout layout {};
		if (!TileGetTextureBlockLayout(ImageOps::RenderTargetTransferFormat(format.bytes_per_element),
		                                cb.attrib3.tile_mode, false, layout) ||
		    (cb.base.addr & (layout.block.block_size - 1u)) != 0) {
			return reject(DepthCopyFallback::TargetTiling);
		}
	}
	std::array<Prospero::ColorComponentMapping, 8> mappings {};
	mappings[0] = format.export_mapping;
	ShaderPixelInputInfo pixel_info {};
	const auto pixel = PrepareProgram(ps, sh, mappings, pixel_info);
	if (pixel.code.size() < depth_copy_pixel_code.size() ||
	    !std::equal(depth_copy_pixel_code.begin(), depth_copy_pixel_code.end(), pixel.code.begin())) {
		return reject(DepthCopyFallback::PixelShader);
	}
	DepthStencilColorCopyPlan plan {};
	plan.aspect = stencil ? vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlagBits::eDepth;
	plan.color_format = stencil ? vk::Format::eR8Uint : vk::Format::eR32Sfloat;
	plan.region = {{scissor.left, scissor.top},
	               {static_cast<uint32_t>(scissor.right - scissor.left),
	                static_cast<uint32_t>(scissor.bottom - scissor.top)}};
	plan.color_info.data.address = cb.base.addr;
	plan.color_info.pixel_format = format.format;
	plan.color_info.guest_format = format.guest_format;
	plan.color_info.type = Prospero::ImageType::kColor2D;
	plan.color_info.extent = {width, height, 1};
	plan.color_info.resources = {1, 1};
	plan.color_info.bytes_per_block = format.bytes_per_element;
	plan.color_info.samples = 1;
	plan.color_info.tile_mode = cb.attrib3.tile_mode;
	plan.color_info.UpdateSize();
	return plan;
}

} // namespace

bool RenderExecutor::DepthStencilColorCopy(CommandBuffer& buffer, const DrawAutoArgs& args) {
	const auto& hw = buffer.GetRegisters();
	const auto& rc = hw.GetRenderControl();
	if (!rc.copy_depth_to_color && !rc.copy_stencil_to_color) {
		return false;
	}
	TraceDepthCopyState(buffer);
	DepthCopyFallback reason {};
	const auto plan = PlanDepthStencilColorCopy(buffer, args, reason);
	if (!plan) {
		TraceDepthCopyFallback(buffer, reason);
		return false;
	}
	auto source_desc = MakeDepthTargetDesc(buffer, hw.GetDepthRenderTarget());
	const auto overlaps_color = [&](GuestRange range) {
		return !range.Empty() && ImageRangeOverlaps(range.address, range.size,
		                                           plan->color_info.data.address, plan->color_info.data.size);
	};
	if (!plan->color_info.data.Valid() || overlaps_color(source_desc.info.data) ||
	    overlaps_color(source_desc.info.stencil) || overlaps_color(source_desc.info.metadata.range)) {
		TraceDepthCopyFallback(buffer, DepthCopyFallback::AliasedRanges);
		return false;
	}
	auto& cache = m_context.GetTextureCache();
	const auto source_id = cache.FindImage(source_desc, true);
	BindRenderTarget(source_id);
	RenderColorInfo color {};
	ResolveRenderColorTarget(buffer, color, 0, 0, false, true);
	if (!color.image_id || cache.m_slot_images.try_get(source_id) == nullptr ||
	    cache.m_slot_images.try_get(color.image_id) == nullptr) {
		TraceDepthCopyFallback(buffer, DepthCopyFallback::Discovery);
		ResetBindings();
		return false;
	}
	auto& source = cache.GetImage(source_id);
	auto& destination = cache.GetImage(color.image_id);
	auto& scratch = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto& source_view = source_desc.view_info;
	const auto& color_view = color.desc.view_info;
	const auto native_fits = [&](const Image& image, const ImageViewInfo& view) {
		if (view.base_level >= image.backing.mip_levels || view.base_layer >= image.backing.layers) {
			return false;
		}
		return static_cast<uint64_t>(plan->region.offset.x) + plan->region.extent.width <=
		           std::max(image.backing.extent.width >> view.base_level, 1u) &&
		       static_cast<uint64_t>(plan->region.offset.y) + plan->region.extent.height <=
		           std::max(image.backing.extent.height >> view.base_level, 1u);
	};
	const uint32_t bytes_per_texel = plan->aspect == vk::ImageAspectFlagBits::eDepth ? 4u : 1u;
	if (source_id == color.image_id || source.backing.image == nullptr ||
	    destination.backing.image == nullptr || source.backing.image == destination.backing.image ||
	    source.depth_id || destination.depth_id || source.binding.needs_rebind ||
	    destination.binding.needs_rebind || source.backing.samples != 1 || destination.backing.samples != 1 ||
	    source.backing.image_type != vk::ImageType::e2D || destination.backing.image_type != vk::ImageType::e2D ||
	    (source.backing.format != vk::Format::eD32SfloatS8Uint &&
	     (plan->aspect != vk::ImageAspectFlagBits::eDepth || source.backing.format != vk::Format::eD32Sfloat)) ||
	    destination.backing.format != plan->color_format || source_view.base_level != 0 ||
	    source_view.base_layer != 0 || source_view.level_count != 1 || source_view.layer_count != 1 ||
	    color_view.level_count != 1 || color_view.layer_count != 1 ||
	    !native_fits(source, source_view) || !native_fits(destination, color_view) ||
	    scratch.Handle() == nullptr ||
	    static_cast<uint64_t>(plan->region.extent.width) * bytes_per_texel > scratch.Size()) {
		TraceDepthCopyFallback(buffer, DepthCopyFallback::NativeBacking);
		ResetBindings();
		return false;
	}
	// Every condition that can reject the low-level copy has now been checked.
	// No fallback is allowed after consuming a deferred clear.
	cache.UpdateImage(color.image_id);
	const bool depth_clear = cache.PrepareDepthCopySource(
	    buffer, source_id, source_desc, plan->aspect, hw.GetDepthClearValue());
	vk::ImageCopy region {};
	region.srcSubresource = {plan->aspect, source_view.base_level, source_view.base_layer, 1};
	region.dstSubresource = {vk::ImageAspectFlagBits::eColor, color_view.base_level, color_view.base_layer, 1};
	region.srcOffset = {plan->region.offset.x, plan->region.offset.y, 0};
	region.dstOffset = region.srcOffset;
	region.extent = {plan->region.extent.width, plan->region.extent.height, 1};
	EXIT_IF(!destination.CopyDepthStencilToColor(source, scratch, region));
	{
		std::scoped_lock lock {cache.m_lock};
		destination.usage.render_target = true;
		cache.TrackImage(color.image_id);
		cache.CommitGpuWrite(destination);
		cache.TrackImageDownload(color.image_id, destination);
	}
	if (DepthCopyTraceEnabled()) {
		std::array<uint64_t, 6> key {static_cast<uint32_t>(plan->aspect),
		    (static_cast<uint64_t>(region.extent.width) << 32u) | region.extent.height,
		    (static_cast<uint64_t>(region.srcOffset.x) << 32u) | static_cast<uint32_t>(region.srcOffset.y),
		    static_cast<uint32_t>(source.backing.format), static_cast<uint32_t>(plan->color_format),
		    static_cast<uint64_t>(depth_clear)};
		static std::mutex mutex;
		static std::array<decltype(key), 32> configurations {};
		static uint32_t count = 0;
		const std::lock_guard lock(mutex);
		const auto end = configurations.begin() + count;
		if (count < configurations.size() && std::find(configurations.begin(), end, key) == end) {
			configurations[count++] = key;
			LOGF("DepthCopy: expanded %s to %s source=0x%016" PRIx64 " color=0x%016" PRIx64
			     " extent=%ux%u region=(%d,%d)+%ux%u htile_clear=%u\n",
			     plan->aspect == vk::ImageAspectFlagBits::eDepth ? "depth" : "stencil",
			     vk::to_string(plan->color_format).c_str(),
			     plan->aspect == vk::ImageAspectFlagBits::eDepth ? source_desc.info.data.address : source_desc.info.stencil.address,
			     color.desc.info.data.address, source_desc.info.extent.width, source_desc.info.extent.height,
			     region.srcOffset.x, region.srcOffset.y, region.extent.width, region.extent.height,
			     static_cast<unsigned>(depth_clear));
		}
	}
	return true;
}

bool RenderExecutor::DepthStencilCopy(CommandBuffer& buffer) {
	const auto& hw       = buffer.GetRegisters();
	const auto& z        = hw.GetDepthRenderTarget();
	const auto& override = hw.GetDepthRenderOverride();
	if (hw.GetColorControl().mode != 0) {
		return false;
	}
	const bool depth_copy = override.force_z_dirty && override.force_z_valid &&
	                        z.z_info.format != Prospero::DepthFormat::kInvalid &&
	                        z.z_read_base_addr != 0 && z.z_write_base_addr != 0 &&
	                        z.z_read_base_addr != z.z_write_base_addr;
	const bool stencil_copy = override.force_stencil_dirty && override.force_stencil_valid &&
	                          z.stencil_info.format != Prospero::StencilFormat::kInvalid &&
	                          z.stencil_read_base_addr != 0 && z.stencil_write_base_addr != 0 &&
	                          z.stencil_read_base_addr != z.stencil_write_base_addr;
	if (!depth_copy && !stencil_copy) {
		return false;
	}

	auto  read_desc  = MakeDepthTargetDesc(buffer, z);
	auto  write_desc = MakeDepthTargetDesc(buffer, z, true);
	auto& cache      = m_context.GetTextureCache();
	const auto read_id = cache.FindImage(read_desc);
	BindRenderTarget(read_id);
	const auto write_id = cache.FindImage(write_desc);
	BindRenderTarget(write_id);
	cache.UpdateImage(read_id);
	cache.UpdateImage(write_id);
	cache.MarkGpuWritten(write_id);
	auto& source      = cache.GetImage(read_id);
	auto& destination = cache.GetImage(write_id);
	EXIT_IF(read_id == write_id || source.backing.format != destination.backing.format);

	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	const ImageSubresourceRange range {read_desc.view_info.base_level, 1,
	                                  read_desc.view_info.base_layer,
	                                  read_desc.view_info.layer_count};
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	               range, command);
	destination.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	                    range, command);
	std::array<vk::ImageCopy, 2> regions {};
	uint32_t                   count = 0;
	for (const auto aspect: {vk::ImageAspectFlagBits::eDepth, vk::ImageAspectFlagBits::eStencil}) {
		if ((aspect == vk::ImageAspectFlagBits::eDepth && !depth_copy) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil && !stencil_copy)) {
			continue;
		}
		auto& region = regions[count++];
		region.srcSubresource = {aspect, range.base_level, range.base_layer,
		                         range.layer_count};
		region.dstSubresource = region.srcSubresource;
		region.extent = write_desc.info.extent;
	}
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
	                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
	                  count, regions.data());
	return true;
}

vk::ImageAspectFlags RenderDepthInfo::AttachmentWriteAspects() const {
	const auto format = desc.view_info.format;
	if (format == vk::Format::eUndefined) {
		return {};
	}

	const auto           available = ImageViewOps::DepthAspectMask(format);
	vk::ImageAspectFlags writes {};
	if ((available & vk::ImageAspectFlagBits::eDepth) &&
	    (depth_load_clear_enable || depth_write_enable)) {
		writes |= vk::ImageAspectFlagBits::eDepth;
	}
	if (!(available & vk::ImageAspectFlagBits::eStencil)) {
		return writes;
	}

	const auto face_writes = [&](const vk::StencilOpState& state) {
		if (state.writeMask == 0) {
			return false;
		}
		bool can_pass = state.compareOp != vk::CompareOp::eNever;
		bool can_fail = state.compareOp != vk::CompareOp::eAlways;
		if (state.compareMask == 0) {
			switch (state.compareOp) {
				case vk::CompareOp::eEqual:
				case vk::CompareOp::eLessOrEqual:
				case vk::CompareOp::eGreaterOrEqual:
				case vk::CompareOp::eAlways:
					can_pass = true;
					can_fail = false;
					break;
				case vk::CompareOp::eNever:
				case vk::CompareOp::eLess:
				case vk::CompareOp::eGreater:
				case vk::CompareOp::eNotEqual:
					can_pass = false;
					can_fail = true;
					break;
				default: break;
			}
		}
		const bool depth_pass = !depth_test_enable || depth_compare_op != vk::CompareOp::eNever;
		const bool depth_fail = depth_test_enable && depth_compare_op != vk::CompareOp::eAlways;
		return (can_fail && state.failOp != vk::StencilOp::eKeep) ||
		       (can_pass && depth_pass && state.passOp != vk::StencilOp::eKeep) ||
		       (can_pass && depth_fail && state.depthFailOp != vk::StencilOp::eKeep);
	};
	if (stencil_clear_enable ||
	    (stencil_test_enable && (face_writes(stencil_front) || face_writes(stencil_back)))) {
		writes |= vk::ImageAspectFlagBits::eStencil;
	}
	return writes;
}

} // namespace Libs::Graphics
