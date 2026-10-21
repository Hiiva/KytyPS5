// Pure, offline tests: these never launch a guest executable or touch saves.
#include "libs/horizonCalibrationPatch.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <span>
#include <string_view>

namespace {
using namespace Libs::HorizonCalibrationPatch;

void Check(bool ok, const char* message) {
	if (!ok) {
		std::fprintf(stderr, "HorizonCalibrationPatchTests: %s\n", message);
		std::abort();
	}
}

std::array<uint8_t, 10> EventBody(const Edit& edit) {
	return {0x1d, edit.original[0], edit.original[1], edit.original[2],
	        edit.original[3], 0x16, 0x04, 0x0b, 0x53, 0x00};
}

void Parse() {
	const auto shorthand = ParsePlan(" seated ");
	Check(shorthand.valid && shorthand.edits.size() == 1, "invalid shorthand");
	Check(shorthand.edits.front().event == "begin", "shorthand must redirect BeginPlay");
	Check(shorthand.edits.front().file_offset == 75823936563ull,
	      "BeginPlay pak offset differs from known package");
	Check(shorthand.edits.front().original == EntryBytes(0x339b),
	      "unexpected BeginPlay bytecode entry");
	Check(shorthand.edits.front().replacement == EntryBytes(0x50),
	      "seated redirect did not resolve");

	const auto named = ParsePlan("begin=startgame, fadein=seated, intro=0x0055");
	Check(named.valid && named.edits.size() == 3, "invalid multi-event patch");
	Check(named.edits[0].replacement == EntryBytes(0x4fb7), "StartGame not resolved");
	Check(named.edits[1].replacement == EntryBytes(0x0050), "Seated not resolved");
	Check(named.edits[2].replacement == EntryBytes(0x0055), "hex target not resolved");
	Check(named.edits[1].file_offset == 75823939412ull,
	      "fadein pak offset differs from known package");
	Check(ParsePlan("").valid && ParsePlan("").edits.empty(),
	      "empty plan should be a no-op");
	Check(ParsePlan("begin=seated,begin=standing").edits.empty(),
	      "duplicate events should not partially apply");
	Check(!ParsePlan("begin=seated,begin=standing").valid,
	      "duplicate events must invalidate entire plan");
	Check(!ParsePlan("begin=wrong").valid, "unknown target accepted");
	Check(!ParsePlan("bogus=seated").valid, "unknown event accepted");
	Check(!ParsePlan("begin=seated,").valid, "trailing comma accepted");
	Check(!ParsePlan("begin=0x8000").valid, "out-of-range graph offset accepted");
	Check(!ParsePlan("begin=-1").valid, "negative graph offset accepted");
	Check(!ParsePlan("begin=0x12ffgarbage").valid, "numeric suffix accepted");
	Check(!ParsePlan("begin=").valid, "empty graph target accepted");
}

void ApplyCompleteBlock() {
	const auto plan = ParsePlan("begin=startgame");
	Check(plan.valid && plan.edits.size() == 1, "complete-block test plan invalid");
	const auto& edit = plan.edits.front();
	auto body = EventBody(edit);
	const auto backup = body;
	Check(Apply(edit, edit.file_offset - 1, body) == Result::Patched,
	      "valid entry bytes were not redirected");
	Check(body[0] == 0x1d && body[5] == 0x16 && body[6] == 0x04 &&
	          body[7] == 0x0b && body[8] == 0x53 && body[9] == 0x00,
	      "the bytecode signature was modified");
	for (size_t i = 0; i < 4; ++i) {
		Check(body[i + 1] == edit.replacement[i], "not all entry bytes were redirected");
	}
	Check(Apply(edit, edit.file_offset - 1, body) == Result::Unchanged,
	      "already-patched block should be idempotent");
	Check(Apply(edit, edit.file_offset - 100, body) == Result::NoOverlap,
	      "unrelated block was considered a patch target");
	Check(Apply(edit, edit.file_offset - 1, std::span<uint8_t> {}) == Result::NoOverlap,
	      "empty block should be ignored");
	Check(backup != body, "test did not exercise any mutation");
}

void RejectUnexpectedBytes() {
	const auto edit = ParsePlan("begin=startgame").edits.front();
	auto bytes = EventBody(edit);
	bytes[3] = 0x9c;
	const auto original = bytes;
	Check(Apply(edit, edit.file_offset - 1, bytes) == Result::Mismatch,
	      "byte mismatch was accepted");
	Check(bytes == original, "partial changes were made before mismatch detection");
	bytes = EventBody(edit);
	bytes[0] = 0x11;
	Check(Apply(edit, edit.file_offset - 1, bytes) == Result::Mismatch,
	      "wrong opcode was accepted");
	bytes = EventBody(edit);
	bytes[5] = 0x17;
	Check(Apply(edit, edit.file_offset - 1, bytes) == Result::Mismatch,
	      "wrong UFunction trailer was accepted");
	bytes = EventBody(edit);
	bytes[9] = 0x41;
	Check(Apply(edit, edit.file_offset - 1, bytes) == Result::Mismatch,
	      "unexpected post-function byte accepted");
}

void SplitReads() {
	const auto edit = ParsePlan("begin=startgame").edits.front();
	auto body = EventBody(edit);
	// Emulate an APR read chunk ending in the middle of the 4-byte graph offset.
	Check(Apply(edit, edit.file_offset - 1, std::span(body.data(), size_t {2})) ==
	          Result::Patched,
	      "first partial read did not redirect its overlapping bytes");
	Check(Apply(edit, edit.file_offset + 1, std::span(body.data() + 2, size_t {8})) ==
	          Result::Patched,
	      "second partial read did not redirect remaining bytes");
	for (size_t i = 0; i < 4; ++i) {
		Check(body[i + 1] == edit.replacement[i], "split read patch is incomplete");
	}
}

void ParseEveryExport() {
	for (const auto& event : kEvents) {
		const auto plan = ParsePlan(event.name);
		Check(plan.valid && plan.edits.size() == 1, "known graph event rejected");
		Check(plan.edits.front().replacement == EntryBytes(event.graph_entry),
		      "event dispatch offsets are inconsistent");
		Check(plan.edits.front().file_offset >= kUexpBase &&
		          plan.edits.front().file_offset < kUexpBase + 250000,
		      "known UFunction patch is outside target uexp");
	}
}
uint32_t ReadKismetU32(std::span<const uint8_t> bytes, size_t at) {
    Check(at + 4 <= bytes.size(), "truncated Kismet integer");
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) value |= uint32_t(bytes[at + i]) << (i * 8);
    return value;
}

// Execute this branch's small opcode subset using deserialized instruction
// lengths, so the false path follows the actual absolute Kismet jump target.
uint32_t ActivationBranchCall(std::span<const uint8_t> bytes, bool activate) {
    size_t at = 0;
    uint32_t logical = 0x4f8b;
    uint32_t destination = 0;
    uint32_t called = 0;
    while (at < bytes.size()) {
        const auto opcode = bytes[at];
        if (opcode == 0x07) { // EX_JumpIfNot plus a resolved local bool property.
            Check(at + 22 <= bytes.size() && bytes[at + 5] == 0x00,
                  "activation condition must read the original local bool");
            if (!activate) destination = ReadKismetU32(bytes, at + 1);
            at += 22;
            logical += 14; // opcode + u32 target + opcode + FProperty*.
            continue;
        }
        if (logical == destination) destination = 0;
        if (opcode == 0x45) { // EX_LocalVirtualFunction, FScriptName, no arguments.
            Check(at + 10 <= bytes.size() && bytes[at + 9] == 0x16,
                  "activation branch must call a no-argument function");
            if (destination == 0) called = ReadKismetU32(bytes, at + 1);
            at += 10;
            logical += 14;
        } else if (opcode == 0x4d) { // EX_PopExecutionFlow.
            if (destination == 0) return called;
            ++at;
            ++logical;
        } else {
            Check(false, "unexpected activation branch opcode");
        }
    }
    Check(false, "activation branch did not return");
    return 0;
}

void AutoStartOnEndActivation() {
    constexpr std::array<uint8_t, 44> original {
        0x07, 0xa8, 0x4f, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x02, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x45, 0x52, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x4d,
        0x45, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x4d
    };
    constexpr uint64_t offset = 75823927870ull;
    Check(kAutoStartOnEndActivationOffset == offset, "activation condition pak offset changed");
    Check(ActivationBranchCall(original, true) == 0x52,
          "original activation must call AnimateOnEndOfCalibration");
    Check(ActivationBranchCall(original, false) == 0x50,
          "original deactivation must call AnimateOffEndOfCalibration");

    std::array<uint8_t, 46> body {};
    body.front() = 0x16;
    body.back() = 0x1c;
    std::ranges::copy(original, body.begin() + 1);
    const auto before = body;
    Check(ApplyAutoStartOnEndActivation(offset - 1, body) == Result::Patched,
          "activation true branch was not patched");
    const auto patched = body;
    for (size_t i = 0; i < body.size(); ++i) {
        const auto expected = i == 24 ? 0x08 : i == 25 ? 0x03 : before[i];
        Check(body[i] == expected, "AutoEnd changed bytes outside the true call's function name");
    }
    const auto branch = std::span<const uint8_t>(body).subspan(1, original.size());
    Check(ActivationBranchCall(branch, true) == 0x308,
          "activation must dispatch StartGame");
    Check(ActivationBranchCall(branch, false) == 0x50,
          "deactivation must preserve AnimateOffEndOfCalibration, preventing StartGame recursion");
    Check(ApplyAutoStartOnEndActivation(offset - 1, body) == Result::Unchanged && body == patched,
          "conditional AutoEnd patch is not idempotent");

    for (size_t length = 0; length < original.size(); ++length) {
        auto partial = original;
        Check(ApplyAutoStartOnEndActivation(offset, std::span(partial).first(length)) ==
                  Result::NoOverlap, "partial activation condition was patched");
        Check(partial == original, "partial activation read mutated its buffer");
    }
    auto suffix = original;
    Check(ApplyAutoStartOnEndActivation(offset + 1, std::span(suffix).subspan(1)) ==
              Result::NoOverlap && suffix == original, "activation suffix read was modified");
    Check(ApplyAutoStartOnEndActivation(offset - original.size(), suffix) ==
              Result::NoOverlap, "non-overlapping activation read was patched");
    for (size_t i = 0; i < original.size(); ++i) {
        auto invalid = original;
        invalid[i] ^= 0x80;
        const auto corrupt = invalid;
        Check(ApplyAutoStartOnEndActivation(offset, invalid) == Result::Mismatch,
              "activation patch accepted a corrupted signature byte");
        Check(invalid == corrupt, "activation mismatch changed its buffer");
    }
    auto mixed = original;
    mixed[23] = 0x08;
    const auto incomplete = mixed;
    Check(ApplyAutoStartOnEndActivation(offset, mixed) == Result::Mismatch && mixed == incomplete,
          "activation patch accepted a partly replaced function name");
    std::array<uint8_t, 10> wrapper {0x1d, 0xb6, 0x4e, 0, 0, 0x16, 0x04, 0x0b, 0x53, 0};
    const auto original_wrapper = wrapper;
    Check(ApplyAutoStartOnEndActivation(75823934777ull, wrapper) == Result::NoOverlap &&
              wrapper == original_wrapper, "AutoEnd must preserve the activation wrapper entry");
}
void AutoStandardAfterSetup() {
    constexpr std::array<uint8_t, 11> original {
        0x45, 0xc9, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x27, 0x16
    };
    constexpr std::array<uint8_t, 11> replacement {
        0x45, 0x88, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x0b
    };
    constexpr uint64_t offset = 75826518121ull;
    Check(kMovementStandardAfterSetupOffset == offset,
          "post-setup Standard call pak offset changed");
    std::array<uint8_t, 13> body {};
    body.front() = 0x16;
    body.back() = 0x4d;
    std::ranges::copy(original, body.begin() + 1);
    const auto before = body;
    Check(ApplyMovementStandardAfterSetup(offset - 1, body) == Result::Patched,
          "post-setup Standard call did not apply");
    Check(std::ranges::equal(std::span(body).subspan(1, 11), replacement),
          "post-setup Standard call has incorrect bytecode");
    Check(body.front() == before.front() && body.back() == before.back(),
          "post-setup Standard call changed adjacent bytecode");
    const auto patched = body;
    Check(ApplyMovementStandardAfterSetup(offset - 1, body) == Result::Unchanged,
          "post-setup Standard call was not idempotent");
    Check(body == patched, "idempotent post-setup call changed its buffer");

    for (size_t length = 0; length < original.size(); ++length) {
        auto partial = original;
        Check(ApplyMovementStandardAfterSetup(offset, std::span(partial).first(length)) ==
                  Result::NoOverlap, "partial post-setup call was patched");
        Check(partial == original, "partial post-setup read mutated its buffer");
    }
    auto partial = original;
    Check(ApplyMovementStandardAfterSetup(offset + 1, std::span(partial).subspan(1)) ==
              Result::NoOverlap, "post-setup suffix read was patched alone");
    Check(partial == original, "post-setup suffix read mutated its buffer");
    Check(ApplyMovementStandardAfterSetup(offset - original.size(), partial) ==
              Result::NoOverlap, "non-overlapping post-setup read was patched");

    for (size_t i = 0; i < original.size(); ++i) {
        auto invalid = original;
        invalid[i] ^= 0x80;
        const auto corrupt = invalid;
        Check(ApplyMovementStandardAfterSetup(offset, invalid) == Result::Mismatch,
              "post-setup call accepted a corrupted original byte");
        Check(invalid == corrupt, "post-setup mismatch changed its buffer");
    }
    auto mixed = original;
    mixed[1] = replacement[1];
    const auto incomplete = mixed;
    Check(ApplyMovementStandardAfterSetup(offset, mixed) == Result::Mismatch,
          "post-setup call accepted a partly patched signature");
    Check(mixed == incomplete, "partly patched post-setup signature was modified");
}
void AutoMovementChain() {
    std::array<uint8_t, 7> jump {0x06u, 0x5cu, 0x0du, 0, 0, 0x45u, 0};
    const auto original = jump;
    Check(ApplyMovementChoiceJump(kMovementChoiceJump, std::span(jump).first(5)) ==
          Result::NoOverlap, "partial movement jump read was modified");
    Check(jump == original, "partial movement read mutated");
    jump[5] = 0x00;
    Check(ApplyMovementChoiceJump(kMovementChoiceJump, jump) == Result::Mismatch,
          "movement jump accepted unexpected following opcode");
    jump[5] = 0x45;
    Check(ApplyMovementChoiceJump(kMovementChoiceJump, jump) == Result::Patched,
          "movement jump was not redirected");
    Check(jump[1] == 0x81 && jump[2] == 0x1a, "wrong movement jump destination");
    Check(ApplyMovementChoiceJump(kMovementChoiceJump, jump) == Result::Unchanged,
          "movement jump not idempotent");
}
void MainMenuDefaultMap() {
    const auto n = kDefaultMapOriginal.size();
    Check(n == kDefaultMapMainMenu.size(), "map override changes byte length");
    std::array<uint8_t, 80> bytes {};
    std::ranges::copy(kDefaultMapOriginal, bytes.begin() + 8u);
    const uint64_t start = kDefaultMapPakOffset - 8u;
    Check(ApplyMainMenuMap(start, bytes) == Result::Patched,
          "valid packaged DefaultEngine.ini map was not replaced");
    const std::string_view new_map(
        reinterpret_cast<const char*>(bytes.data() + 8u), n);
    Check(new_map == kDefaultMapMainMenu, "main menu map contents differ");
    Check(ApplyMainMenuMap(start, bytes) == Result::Unchanged,
          "main menu override not idempotent");
    Check(ApplyMainMenuMap(start + 100u, bytes) == Result::NoOverlap,
          "main menu override touched unrelated pak data");
    std::ranges::copy(kDefaultMapOriginal, bytes.begin() + 8u);
    bytes[8] = 'X';
    const auto corrupt = bytes;
    Check(ApplyMainMenuMap(start, bytes) == Result::Mismatch,
          "map override accepted corrupt original bytes");
    Check(bytes == corrupt, "map override mutated unexpected data");
    Check(ApplyMainMenuMap(kDefaultMapPakOffset - 3u,
                           std::span(bytes.data(), 5u)) == Result::NoOverlap,
          "map override accepted an incomplete APR chunk");
}
void AutoSaveComfortWidget() {
    const auto& edit = kAutoSaveComfortEdit;
    Check(edit.file_offset == 75824259426ull, "comfort widget pak offset changed");
    Check(edit.original == EntryBytes(0x06d3u), "OnSubMenuCreated entry mismatch");
    Check(edit.replacement == EntryBytes(0x02d6u), "OnSaveButtonClicked entry mismatch");
    auto bytes = EventBody(edit);
    Check(Apply(edit, edit.file_offset - 1u, bytes) == Result::Patched,
          "comfort widget creation was not redirected to Save");
    Check(bytes[1] == 0xd6u && bytes[2] == 0x02u,
          "comfort Save graph entry was not written");
    Check(Apply(edit, edit.file_offset - 1u, bytes) == Result::Unchanged,
          "comfort widget patch is not idempotent");
}
} // namespace

int main() {
	Parse();
	ApplyCompleteBlock();
	RejectUnexpectedBytes();
	SplitReads();
	ParseEveryExport();
	AutoSaveComfortWidget();
	MainMenuDefaultMap();
	AutoStartOnEndActivation();
	AutoStandardAfterSetup();
	AutoMovementChain();
	std::puts("HorizonCalibrationPatchTests: passed");
	return 0;
}
