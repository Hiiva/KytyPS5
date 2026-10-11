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
	std::puts("HorizonCalibrationPatchTests: passed");
	return 0;
}
