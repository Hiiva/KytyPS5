#pragma once

// Desktop tracking input in metres relative to the initial headset position.
// This is an input workspace, not a native tracker limit or a game fade policy.

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <system_error>

namespace Libs::VirtualHeadPosition {

using Offset = std::array<float, 3>;
inline constexpr float kWorkspaceMetres = 0.75f;
inline constexpr std::size_t kMaxScriptSteps = 16;
inline constexpr std::uint32_t kMaxScriptTimeMs = 1800000;
inline constexpr float kMaxYawRadians = 3.14159265359f;
inline constexpr float kMaxPitchRadians = 1.48353f;

struct Step {
    std::uint32_t time_ms = 0;
    Offset offset {};
    float yaw = 0.0f;
    float pitch = 0.0f;
};

struct Script {
    std::array<Step, kMaxScriptSteps> steps {};
    std::size_t count = 0;
};

[[nodiscard]] inline bool IsValidOffset(const Offset& offset) {
    return std::all_of(offset.begin(), offset.end(), [](float value) {
        return std::isfinite(value) && std::abs(value) <= kWorkspaceMetres;
    });
}

// Strict finite sequence: time_ms:x:y:z:yaw:pitch;... . Coordinates are tracker
// space offsets in metres; angles are radians. The final input persists.
[[nodiscard]] inline std::optional<Script> ParseScript(std::string_view text) {
    if (text.size() > 1024) {
        return std::nullopt;
    }
    Script script;
    while (!text.empty()) {
        if (script.count == script.steps.size()) {
            return std::nullopt;
        }
        const auto separator = text.find(';');
        auto token = text.substr(0, separator);
        std::array<std::string_view, 6> fields;
        for (std::size_t field = 0; field < 5; ++field) {
            const auto colon = token.find(':');
            if (colon == std::string_view::npos) {
                return std::nullopt;
            }
            fields[field] = token.substr(0, colon);
            token.remove_prefix(colon + 1);
        }
        fields[5] = token;
        if (std::any_of(fields.begin(), fields.end(), [](auto field) { return field.empty(); })) {
            return std::nullopt;
        }
        Step step;
        const auto time = std::from_chars(fields[0].data(), fields[0].data() + fields[0].size(), step.time_ms);
        if (time.ec != std::errc {} || time.ptr != fields[0].data() + fields[0].size() ||
            step.time_ms > kMaxScriptTimeMs ||
            (script.count != 0 && step.time_ms <= script.steps[script.count - 1].time_ms)) {
            return std::nullopt;
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const auto field = fields[axis + 1];
            const auto value = std::from_chars(field.data(), field.data() + field.size(), step.offset[axis]);
            if (value.ec != std::errc {} || value.ptr != field.data() + field.size()) {
                return std::nullopt;
            }
        }
        if (!IsValidOffset(step.offset)) {
            return std::nullopt;
        }
        const std::array<float*, 2> angles {&step.yaw, &step.pitch};
        const std::array<float, 2> angle_limits {kMaxYawRadians, kMaxPitchRadians};
        for (std::size_t axis = 0; axis < angles.size(); ++axis) {
            const auto field = fields[axis + 4];
            const auto value = std::from_chars(field.data(), field.data() + field.size(), *angles[axis]);
            if (value.ec != std::errc {} || value.ptr != field.data() + field.size() ||
                !std::isfinite(*angles[axis]) || std::abs(*angles[axis]) > angle_limits[axis]) {
                return std::nullopt;
            }
        }
        script.steps[script.count++] = step;
        if (separator == std::string_view::npos) {
            break;
        }
        text.remove_prefix(separator + 1);
        if (text.empty()) {
            return std::nullopt;
        }
    }
    return script;
}

struct State {
    Offset offset {};
    std::size_t next_step = 0;

    // Reject a nonfinite input without changing any axis. Use double for the
    // addition so even a large finite mouse delta can be clamped safely.
    [[nodiscard]] bool Move(const Offset& delta) {
        if (!IsValidOffset(offset) ||
            !std::all_of(delta.begin(), delta.end(), [](float value) { return std::isfinite(value); })) {
            return false;
        }
        for (std::size_t axis = 0; axis < offset.size(); ++axis) {
            offset[axis] = static_cast<float>(std::clamp(
                static_cast<double>(offset[axis]) + delta[axis],
                -static_cast<double>(kWorkspaceMetres), static_cast<double>(kWorkspaceMetres)));
        }
        return true;
    }

    // Apply each scheduled input once. Manual movement or a reset after the
    // final step remains authoritative; an old script step is never replayed.
    std::size_t ApplyDue(const Script& script, std::uint64_t elapsed_ms) {
        const auto previous = next_step;
        while (next_step < script.count && next_step < script.steps.size() &&
               script.steps[next_step].time_ms <= elapsed_ms) {
            offset = script.steps[next_step++].offset;
        }
        return next_step - previous;
    }

    void ResetPosition() { offset = {}; }

    // F9 should return to the most recently applied scripted pose, not erase
    // a held offset that the game still needs for an unobstructed headset.
    // With no applied step, preserve the ordinary neutral reset behavior.
    [[nodiscard]] const Step* ResetToLastAppliedStep(const Script& script) {
        const Step* last = next_step != 0 && next_step <= script.count
                               ? &script.steps[next_step - 1]
                               : nullptr;
        offset = last != nullptr ? last->offset : Offset {};
        return last;
    }
};

} // namespace Libs::VirtualHeadPosition
