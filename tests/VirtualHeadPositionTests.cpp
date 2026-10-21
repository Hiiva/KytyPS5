// Desktop head input tests: finite workspace, held input, timing and SDK poses.
#include "libs/virtualHeadPosition.h"
#include "libs/hmd2TrackerPose.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace {
using namespace Libs::VirtualHeadPosition;

void Verify(bool ok, const char* expression, int line) {
    if (!ok) {
        std::fprintf(stderr, "VirtualHeadPositionTests: line %d failed: %s\n", line, expression);
        std::abort();
    }
}
#define VERIFY(expr) Verify(static_cast<bool>(expr), #expr, __LINE__)

void Near(float actual, float expected) {
    VERIFY(std::isfinite(actual) && std::abs(actual - expected) <= 1.0e-6f);
}

void ParseFiniteSequence() {
    const auto disabled = ParseScript("");
    VERIFY(disabled && disabled->count == 0);
    const auto planned = ParseScript("600000:0:0:-0.05:0.08:0");
    VERIFY(planned && planned->count == 1);
    const auto& step = planned->steps[0];
    VERIFY(step.time_ms == 600000);
    Near(step.offset[0], 0.0f);
    Near(step.offset[1], 0.0f);
    Near(step.offset[2], -0.05f);
    Near(step.yaw, 0.08f);
    Near(step.pitch, 0.0f);

    const auto limits = ParseScript("0:-0.75:0.75:0:-3.14159265359:1.48353;1800000:0:0:5e-2:0:0");
    VERIFY(limits && limits->count == 2);
    VERIFY(limits->steps[0].time_ms == 0 && limits->steps[1].time_ms == kMaxScriptTimeMs);
    Near(limits->steps[0].offset[0], -kWorkspaceMetres);
    Near(limits->steps[0].offset[1], kWorkspaceMetres);
    Near(limits->steps[1].offset[2], 0.05f);

    std::string capacity;
    for (std::size_t i = 0; i < kMaxScriptSteps; ++i) {
        if (i != 0) { capacity += ';'; }
        capacity += std::to_string(i) + ":0:0:0:0:0";
    }
    const auto full = ParseScript(capacity);
    VERIFY(full && full->count == kMaxScriptSteps);
    VERIFY(!ParseScript(capacity + ";16:0:0:0:0:0"));
    VERIFY(!ParseScript(std::string(1025, '0')));
}

void RejectMalformedInput() {
    const std::array invalid {
        ";", "0", "0:0:0:0:0", "0:0:0:0:0:0:0", "0::0:0:0:0",
        "0:0:0:0:0:", "0:0:0:0:0:0;", "0:0:0:0:0:0;;1:0:0:0:0:0",
        "-1:0:0:0:0:0", "+1:0:0:0:0:0", "1.0:0:0:0:0:0",
        "1800001:0:0:0:0:0", "4294967296:0:0:0:0:0",
        "0:0:0:0:0:0;0:0:0:0:0:0", "2:0:0:0:0:0;1:0:0:0:0:0",
        " 0:0:0:0:0:0", "0:0:0:0:0:0 ", "0:0:0:0:0:0\n",
        "0:0.751:0:0:0:0", "0:0:-0.751:0:0:0", "0:0:0:1:0:0",
        "0:nan:0:0:0:0", "0:0:inf:0:0:0", "0:0:0:-inf:0:0",
        "0:0:0:0:nan:0", "0:0:0:0:0:inf", "0:0:0:0:3.142:0",
        "0:0:0:0:0:-1.484", "0:1e99:0:0:0:0", "0:0x1:0:0:0:0",
        "0:1f:0:0:0:0", "0:0:0:0:0:1,5"
    };
    for (const auto* text : invalid) {
        VERIFY(!ParseScript(text));
    }
}

void WorkspaceAndAtomicRejection() {
    State state;
    VERIFY(state.Move({0.2f, -0.3f, 0.1f}));
    const auto before = state.offset;
    VERIFY(!state.Move({0.1f, std::numeric_limits<float>::quiet_NaN(), 0.1f}));
    VERIFY(state.offset == before);
    VERIFY(!state.Move({std::numeric_limits<float>::infinity(), 0.0f, 0.0f}));
    VERIFY(state.offset == before);
    VERIFY(state.Move({1.0f, -1.0f, -1.0f}));
    VERIFY((state.offset == Offset {0.75f, -0.75f, -0.75f}));
    const auto huge = std::numeric_limits<float>::max();
    VERIFY(state.Move({-huge, huge, huge}));
    VERIFY((state.offset == Offset {-0.75f, 0.75f, 0.75f}));
    state.offset = {0.0f, 0.8f, 0.0f};
    const auto invalid_before = state.offset;
    VERIFY(!state.Move({0.1f, 0.0f, 0.0f}));
    VERIFY(state.offset == invalid_before);
    state.ResetPosition();
    VERIFY(state.offset == Offset {});
}

void HeldInputAndReset() {
    const auto script = ParseScript("600000:0:0:-0.05:0.08:0;700000:0.1:0.2:0.3:0:0");
    VERIFY(script);
    State state;
    VERIFY(state.ApplyDue(*script, 599999) == 0);
    VERIFY(state.next_step == 0 && state.offset == Offset {});
    VERIFY(state.ApplyDue(*script, 600000) == 1);
    VERIFY(state.next_step == 1);
    Near(state.offset[2], -0.05f);
    VERIFY(state.ApplyDue(*script, 600001) == 0);
    Near(state.offset[2], -0.05f);
    VERIFY(state.Move({0.1f, 0.0f, 0.0f}));
    VERIFY(state.ApplyDue(*script, 699999) == 0);
    Near(state.offset[0], 0.1f);
    state.ResetPosition();
    VERIFY(state.next_step == 1 && state.offset == Offset {});
    // A reset and even a backward clock observation do not replay old input.
    VERIFY(state.ApplyDue(*script, 0) == 0);
    VERIFY(state.ApplyDue(*script, 600000) == 0);
    VERIFY(state.offset == Offset {});
    // The next explicitly scheduled input is still due at its original time.
    VERIFY(state.ApplyDue(*script, 700000) == 1);
    VERIFY((state.offset == Offset {0.1f, 0.2f, 0.3f}));
    state.ResetPosition();
    VERIFY(state.Move({0.0f, 0.0f, 0.04f}));
    VERIFY(state.ApplyDue(*script, std::numeric_limits<std::uint64_t>::max()) == 0);
    Near(state.offset[2], 0.04f);
    State catch_up;
    VERIFY(catch_up.ApplyDue(*script, 700001) == 2);
    VERIFY(catch_up.next_step == 2 && catch_up.offset == script->steps[1].offset);
}

void ResetToLastAppliedStep() {
    const auto script = ParseScript("100:0:0:-0.05:0.08:0;200:0.1:0.2:-0.1:0.16:0.02");
    VERIFY(script && script->count == 2);
    State state;
    VERIFY(state.Move({0.2f, 0.0f, 0.0f}));
    VERIFY(state.ResetToLastAppliedStep(*script) == nullptr);
    VERIFY(state.offset == Offset {} && state.next_step == 0);

    VERIFY(state.ApplyDue(*script, 100) == 1);
    VERIFY(state.Move({0.3f, 0.0f, 0.0f}));
    const Step* first = state.ResetToLastAppliedStep(*script);
    VERIFY(first == &script->steps[0]);
    VERIFY(state.offset == first->offset && state.next_step == 1);
    Near(first->yaw, 0.08f);

    VERIFY(state.ApplyDue(*script, 200) == 1);
    VERIFY(state.Move({0.0f, 0.3f, 0.0f}));
    const Step* second = state.ResetToLastAppliedStep(*script);
    VERIFY(second == &script->steps[1]);
    VERIFY(state.offset == second->offset && state.next_step == 2);
    Near(second->pitch, 0.02f);

    Script empty;
    VERIFY(state.ResetToLastAppliedStep(empty) == nullptr);
    VERIFY(state.offset == Offset {} && state.next_step == 2);
}

float ReadFloat(const std::array<std::uint8_t, Libs::Hmd2TrackerPose::kHeadsetResultBytes>& bytes,
                std::size_t offset) {
    float value = 0.0f;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void ScriptedPoseReachesSdkHeadAndEyes() {
    const auto script = ParseScript("600000:0:0:-0.05:0.08:0");
    VERIFY(script);
    State state;
    VERIFY(state.ApplyDue(*script, 600000) == 1);
    const auto& step = script->steps[0];
    const std::array<float, 3> position {state.offset[0], 1.6f + state.offset[1], state.offset[2]};
    const std::array<float, 4> q {0.0f, std::sin(step.yaw * 0.5f), 0.0f, std::cos(step.yaw * 0.5f)};
    std::array<std::uint8_t, Libs::Hmd2TrackerPose::kHeadsetResultBytes> bytes {};
    VERIFY(Libs::Hmd2TrackerPose::Write(bytes, position, q, 600000000u, true));
    float eye_distance_squared = 0.0f;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        // Both SDK center poses, including Horizon's +0x120 path, share input.
        Near(ReadFloat(bytes, 0x60 + axis * 4), position[axis]);
        Near(ReadFloat(bytes, 0x120 + axis * 4), position[axis]);
        const auto left = ReadFloat(bytes, 0xa0 + axis * 4);
        const auto right = ReadFloat(bytes, 0xe0 + axis * 4);
        Near((left + right) * 0.5f, position[axis]);
        eye_distance_squared += (right - left) * (right - left);
    }
    Near(std::sqrt(eye_distance_squared), Libs::Hmd2TrackerPose::kInterpupillaryDistance);
    for (const std::size_t pose : {0x60u, 0xa0u, 0xe0u, 0x120u}) {
        for (std::size_t axis = 0; axis < q.size(); ++axis) {
            Near(ReadFloat(bytes, pose + 0x10 + axis * 4), q[axis]);
        }
    }
}
} // namespace

int main() {
    ParseFiniteSequence();
    RejectMalformedInput();
    WorkspaceAndAtomicRejection();
    HeldInputAndReset();
    ResetToLastAppliedStep();
    ScriptedPoseReachesSdkHeadAndEyes();
    std::puts("VirtualHeadPositionTests: 6 cases passed");
}
