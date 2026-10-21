#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	DispatchIndirect,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;

	[[nodiscard]] vk::CommandBuffer Handle() const;
	struct DebugSnapshot {
		uint32_t op;
		uint64_t submit_id;
		std::array<uint32_t, 4> args;
		uint64_t arg4;
	};
	[[nodiscard]] DebugSnapshot GetDebugSnapshot() const noexcept {
		return {m_debug_op, m_debug_submit_id,
		        {m_debug_arg0, m_debug_arg1, m_debug_arg2, m_debug_arg3}, m_debug_arg4};
	}
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);

private:
	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool DepthStencilColorCopy(CommandBuffer& buffer, const DrawAutoArgs& args);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	void TracePanelDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                    const DrawEmitInfo& emit, const DrawRenderState& state,
	                    const RenderState& rendering,
	                    std::span<PreparedBindings* const> stages);
	struct PanelCaptureState {
		uint32_t index = 0;
		uint32_t misses = 0;
		uint64_t bytes = 0;
	};
	enum class PanelGpuScalarKind { CommonRgb, HdrUv, CompositeControls, CompositeUv, CompositeGains };
	using PanelGpuScalarWords = std::array<uint32_t, 12>;
	using PanelGpuCompositeControlWords = std::array<uint32_t, 7>;
	using PanelGpuCompositeGainWords = std::array<uint32_t, 2>;
	struct PanelGpuScalarRecipe {
		uint64_t pixel_hash;
		uint32_t resource;
		uint32_t first_pc;
		uint32_t declared_size;
		uint32_t selected_offset;
		uint32_t selected_size;
	};
	[[nodiscard]] static const PanelGpuScalarRecipe* GetPanelGpuScalarRecipe(PanelGpuScalarKind kind);
	struct PanelGpuScalarPlan {
		PanelGpuScalarKind kind = PanelGpuScalarKind::CommonRgb;
		vk::DescriptorBufferInfo native {};
		uint64_t guest_address = 0;
		uint64_t guest_size = 0;
		uint64_t selected_offset = 0;
		uint64_t owner_size = 0;
		std::array<uint32_t, 4> sharp {};
		uint32_t stage_index = 0;
		uint32_t packed_offset_word = 0;
		bool stream = false;
	};
	// Pure descriptor/owner validation. Caller retains GPU-thread serialization.
	[[nodiscard]] const char* PlanPanelGpuScalarCapture(
	    const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	    const RenderState& rendering, std::span<PreparedBindings* const> stages,
	    PanelGpuScalarPlan& plan) const;
	[[nodiscard]] const char* PlanPanelGpuHdrUvCapture(
	    const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	    const RenderState& rendering, std::span<PreparedBindings* const> stages,
	    PanelGpuScalarPlan& plan) const;
	[[nodiscard]] const char* PlanPanelGpuCompositeControlsCapture(
	    const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	    const RenderState& rendering, std::span<PreparedBindings* const> stages,
	    PanelGpuScalarPlan& plan) const;
	[[nodiscard]] const char* PlanPanelGpuCompositeUvCapture(
	    const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	    const RenderState& rendering, std::span<PreparedBindings* const> stages,
	    PanelGpuScalarPlan& plan) const;
	[[nodiscard]] const char* PlanPanelGpuCompositeGainsCapture(
	    const CommandBuffer& buffer, std::span<const RenderColorInfo> colors, ImageId depth,
	    const RenderState& rendering, std::span<PreparedBindings* const> stages,
	    PanelGpuScalarPlan& plan) const;
	[[nodiscard]] const char* PlanPanelGpuCapture(
	    PanelGpuScalarKind kind, const CommandBuffer& buffer, std::span<const RenderColorInfo> colors,
	    ImageId depth, const RenderState& rendering, std::span<PreparedBindings* const> stages,
	    PanelGpuScalarPlan& plan) const;
	// Records a four-byte copy; completion runs only after the normal scheduler tick.
	[[nodiscard]] bool QueuePanelGpuScalarReadback(
	    CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	    Common::UniqueFunction<void, bool, uint32_t>&& completion);
	// Fixed recipes retain their exact selected spans and callback widths.
	[[nodiscard]] bool QueuePanelGpuHdrUvReadback(
	    CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	    Common::UniqueFunction<void, bool, PanelGpuScalarWords>&& completion);
	[[nodiscard]] bool QueuePanelGpuCompositeControlsReadback(
	    CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	    Common::UniqueFunction<void, bool, PanelGpuCompositeControlWords>&& completion);
	[[nodiscard]] bool QueuePanelGpuCompositeUvReadback(
	    CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	    Common::UniqueFunction<void, bool, PanelGpuScalarWords>&& completion);
	[[nodiscard]] bool QueuePanelGpuCompositeGainsReadback(
	    CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	    Common::UniqueFunction<void, bool, PanelGpuCompositeGainWords>&& completion);
	[[nodiscard]] bool QueuePanelGpuReadback(
	    PanelGpuScalarKind kind, CommandBuffer& buffer, const PanelGpuScalarPlan& plan,
	    Common::UniqueFunction<void, bool, PanelGpuScalarWords>&& completion);
	[[nodiscard]] uint32_t CapturePanelGpuScalarBeforeDraw(
	    uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	    const DrawEmitInfo& emit, const DrawRenderState& state, const RenderState& rendering,
	    std::span<PreparedBindings* const> stages);
	[[nodiscard]] uint32_t CapturePanelGpuCompositeBeforeDraw(
	    uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	    const DrawEmitInfo& emit, const DrawRenderState& state, const RenderState& rendering,
	    std::span<PreparedBindings* const> stages);
	struct PanelSourceWatch {
		ImageId image_id;
		vk::Image native = nullptr;
		ImageViewInfo view;
		GuestRange data;
		GuestRange stencil;
		GuestRange metadata;
		uint32_t parent_mask = 0;
		uint32_t last_observed_frame = 0;
	};
	static constexpr uint32_t PanelPrimaryWatchCapacity = 8;
	static constexpr uint32_t PanelUpstreamWatchCapacity = 4;
	static constexpr uint32_t PanelPrimaryWatchMask = (1u << PanelPrimaryWatchCapacity) - 1u;
	static constexpr uint32_t PanelUpstreamWatchMask = ((1u << PanelUpstreamWatchCapacity) - 1u) << PanelPrimaryWatchCapacity;
	struct PanelProducerMatch {
		uint32_t images = 0;
		uint32_t buffer_aliases = 0;
		uint32_t metadata_aliases = 0;
		bool unattributed_writes = false;
		[[nodiscard]] bool Any() const { return (images | buffer_aliases | metadata_aliases) != 0; }
	};
	void TracePanelBindings(const char* label, uint32_t call, CommandBuffer& buffer,
	                        std::string record, std::span<const RenderColorInfo> colors,
	                        ImageId depth, std::span<PreparedBindings* const> stages,
	                        bool rendering_clear, bool trace_due, const std::string* capture_prefix,
	                        uint32_t milestone, PanelCaptureState& capture);
	[[nodiscard]] static bool PanelProducerTraceEnabled();
	void ObservePanelSources(const CommandBuffer& buffer, std::span<PreparedBindings* const> stages);
	void ObservePanelUpstreamSources(const CommandBuffer& buffer, std::span<PreparedBindings* const> stages,
	                                uint32_t parent_mask);
	void LogPanelUpstreamSkip(uint32_t frame, uint32_t reason_bit, const char* reason);
	[[nodiscard]] uint32_t PanelSourceLimit() const {
		return m_upstream_source_count == 0 ? m_panel_source_count : PanelPrimaryWatchCapacity + m_upstream_source_count;
	}
	[[nodiscard]] uint32_t MatchPanelSourceImage(ImageId id, const ImageViewInfo& view) const;
	[[nodiscard]] uint32_t MatchPanelSourceRange(GuestRange range, bool metadata) const;
	[[nodiscard]] PanelProducerMatch MatchPanelProducerWrites(
	    std::span<const RenderColorInfo> colors, std::span<PreparedBindings* const> stages) const;
	[[nodiscard]] bool TracePanelProducerWindow(const CommandBuffer& buffer, uint32_t frame, const PanelProducerMatch& match);
	void TracePanelProducerDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                            const DrawEmitInfo& emit, const DrawRenderState& state,
	                            const RenderState& rendering, std::span<PreparedBindings* const> stages,
	                            uint32_t scalar_capture = 0);
	void TracePanelProducerDispatch(uint64_t submit_id, CommandBuffer& buffer, PreparedBindings& bindings,
	                                std::array<uint32_t, 3> groups, uint32_t mode, bool async_compute,
	                                vk::Buffer indirect_buffer = nullptr, uint64_t indirect_offset = 0);
	void TracePanelProducerShortcut(const CommandBuffer& buffer, const char* operation, GuestRange range,
	                                uint64_t shader_hash, uint32_t value, bool metadata = false,
	                                ImageId image = {}, const ImageViewInfo* view = nullptr);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	uint32_t                             m_panel_trace_draws = 0;
	PanelCaptureState                    m_panel_capture;
	PanelCaptureState                    m_producer_capture;
	PanelCaptureState                    m_gpu_scalar_capture;
	uint32_t                             m_gpu_scalar_decisions = 0;
	uint32_t                             m_gpu_scalar_last_frame = UINT32_MAX;
	std::array<PanelSourceWatch, PanelPrimaryWatchCapacity + PanelUpstreamWatchCapacity> m_panel_sources {};
	uint32_t                             m_panel_source_count = 0;
	uint32_t                             m_panel_source_logs = 0;
	uint32_t                             m_upstream_source_count = 0;
	uint32_t                             m_upstream_watch_logs = 0;
	uint32_t                             m_upstream_skip_reasons = 0;
	uint32_t                             m_upstream_skipped = 0;
	std::array<uint32_t, 8>               m_upstream_discovery_frames {};
	uint32_t                             m_upstream_discovery_count = 0;
	uint32_t                             m_producer_first_frame = UINT32_MAX;
	uint32_t                             m_producer_frame = UINT32_MAX;
	uint32_t                             m_producer_order = 0;
	uint32_t                             m_producer_matches = 0;
	uint32_t                             m_producer_emitted = 0;
	uint32_t                             m_producer_unattributed = 0;
	uint32_t                             m_producer_upstream_native_matches = 0;
	bool                                 m_producer_window = false;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
