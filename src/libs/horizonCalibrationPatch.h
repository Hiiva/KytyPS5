#pragma once

// An opt-in, in-memory patch for Horizon's cooked UE4.26 calibration Blueprint.
// The patch operates on the temporary APR read buffer, never on the game's pak.
// Keep all offsets version-specific and fail closed if the data is unexpected.

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace Libs::HorizonCalibrationPatch {

// PPSA05666, 01.004.011. A different pak size is not eligible for this patch.
inline constexpr uint64_t kPakSize = 84430726574ull;
inline constexpr uint64_t kUexpBase = 75823884858ull;
inline constexpr uint64_t kUassetLength = 66433ull;
inline constexpr uint64_t kEntryOffset = 32ull;

struct Event {
	std::string_view name;
	uint64_t         serialized_offset;
	uint32_t         graph_entry;
};

inline constexpr std::array<Event, 39> kEvents {{
    {"begin",       118106u, 0x339bu},
    {"intro",       118158u, 0x3396u},
    {"afterintro",  118002u, 0x3465u},
    {"fadein",      120955u, 0x15ffu},
    {"fadeout",     120903u, 0x163bu},
    {"seated",      122344u, 0x0050u},
    {"standing",    122292u, 0x0055u},
    {"seatedclicks",122101u, 0x05c0u},
    {"gazeoff",     116720u, 0x465du},
    {"gazeon",      116772u, 0x459fu},
    {"gazefadein",  116668u, 0x4b85u},
    {"gazefadeout", 116616u, 0x4bc1u},
    {"subtitlefadein",117603u, 0x3aacu},
    {"subtitlefadeout",117551u, 0x3ae8u},
    {"subtitleoff", 117655u, 0x3aa0u},
    {"subtitleon",  117707u, 0x3a94u},
    {"dominantfadein",116876u, 0x451du},
    {"dominantfadeout",116824u, 0x4559u},
    {"dominantright",116928u, 0x4510u},
    {"dominantleft",116980u, 0x4503u},
    {"endclick",    116078u, 0x5195u},
    {"onfadeout",   121302u, 0x0e6fu},
    {"startgame",   116182u, 0x4fb7u},
    {"comfortmenu",     121354u, 0x0bc9u},
    {"comfortselected", 118054u, 0x33a0u},
    {"headsetfadein",   120556u, 0x1b9cu},
    {"headsetfadeout",  120504u, 0x1bd8u},
    {"handfadein",      120261u, 0x1df2u},
    {"handfadeout",     120209u, 0x1e2eu},
    {"camerafadein",    119968u, 0x23ddu},
    {"camerafadeout",   119916u, 0x2419u},
    {"reminderfadein",  119812u, 0x2508u},
    {"reminderfadeout", 119760u, 0x257eu},
    {"endforward",      115558u, 0x5edcu},
    {"endbuttonslayout", 115610u, 0x5d73u},
    {"endanimateon",     115974u, 0x594au},
    {"endanimateoff",    115922u, 0x595du},
    {"endtitlefadeout",  115454u, 0x5f79u},
    {"continuewarning",  118262u, 0x3317u},
}};

struct Edit {
	std::string_view     event;
	uint64_t             file_offset;
	std::array<uint8_t, 4> original;
	std::array<uint8_t, 4> replacement;
};

// BP_FrontEndMenu_ComfortSettings_Custom, PPSA05666 01.004.011.
// OnSubMenuCreated delegates to Kismet entry 0x06d3.
// OnSaveButtonClicked delegates to entry 0x02d6 with no arguments.
// Redirect the former to the latter only when explicitly opted in.
inline constexpr Edit kAutoSaveComfortEdit {
    "oncreate", 75824259426ull,
    {0xd3, 0x06, 0x00, 0x00}, {0xd6, 0x02, 0x00, 0x00}
};

// StartupGameMode::ReceiveBeginPlay can take the shipped demo branch's
// OpenLevel(MainMenu) path, without modifying the game's pak on disk.
inline constexpr Edit kStartupMainMenuEdit {
    "startupmainmenu", 11557168072ull,
    {0x95, 0x05, 0x00, 0x00}, {0x31, 0x00, 0x00, 0x00}
};
// BP_MovementPreferenceTutorialMenu (PPSA05666 01.004.011).
// When its panel finishes animating, enter the Standard movement selection
// callback instead of simply finishing the panel animation. Both wrappers
// have the same no-argument dispatch signature.
inline constexpr Edit kMovementAutoStandardEdit {
    "movementstandard", 75826525367ull,
    {0x70, 0x17, 0x00, 0x00}, {0x0b, 0x32, 0x00, 0x00}
};
// Trigger the Standard preset only when the movement preference widget
// reaches FadeInComfortButtons, after mode selection and widget setup.
inline constexpr Edit kMovementConfirmWhenReadyEdit {
    "movementready", 75826523849ull,
    {0xcb, 0x32, 0x00, 0x00}, {0x0b, 0x32, 0x00, 0x00}
};
// OnAnalogueOrGestureClick(bool) -> OnControlPresetButtonClicked broadcaster.
// This callback chain should signal the calibration parent once the widget
// selects its movement style, instead of leaving the submenu awaiting a click.
inline constexpr Edit kMovementPresetNotifyEdit {
    "movementpreset", 75826524335ull,
    {0x96, 0x26, 0x00, 0x00}, {0x51, 0x16, 0x00, 0x00}
};
// OnAnalogueOrGestureClick(bool) normally opens the mode confirmation menu.
// OnAnalogueConfirm is a no-argument event in the same ExecuteUbergraph.
// Compare this with the direct multicast shortcut separately.
inline constexpr Edit kMovementAnalogueConfirmEdit {
    "movementanalogueconfirm", 75826524335ull,
    {0x96, 0x26, 0x00, 0x00}, {0xfe, 0x1f, 0x00, 0x00}
};
// After selecting Analogue, let its click callback accept the Standard
// comfort preset rather than opening the confirmation screen. This is a
// separate experiment from accepting the Analogue movement mode alone.
inline constexpr Edit kMovementStandardFromModeEdit {
    "movementstandardfrommode", 75826524335ull,
    {0x96, 0x26, 0x00, 0x00}, {0x0b, 0x32, 0x00, 0x00}
};
// When the movement preference widget finishes fading out its final child,
// dispatch the parent-facing preset notification. This is deliberately after
// mode and comfort selection, unlike the early animated-panel redirect.
inline constexpr Edit kMovementNotifyAfterFadeEdit {
    "movementnotifyafterfade", 75826523745ull,
    {0x77, 0x33, 0x00, 0x00}, {0x51, 0x16, 0x00, 0x00}
};

enum class Result { NoOverlap, Unchanged, Patched, Mismatch };

// EndOfCalibrationActivation(Activate) is also called with false by CompleteStep.
// Rewriting its wrapper to StartGame recurses through that deactivation callback.
// Keep the wrapper and common widget setup, then replace only the true branch's
// AnimateOnEndOfCalibration() call with StartGame(). The false branch still calls
// AnimateOffEndOfCalibration(). Both calls have the same serialized/runtime size.
inline constexpr uint64_t kAutoStartOnEndActivationOffset = 75823927870ull;
[[nodiscard]] inline Result ApplyAutoStartOnEndActivation(
    uint64_t block_offset, std::span<uint8_t> block) {
    if (block_offset > kAutoStartOnEndActivationOffset ||
        kAutoStartOnEndActivationOffset - block_offset >= block.size()) {
        return Result::NoOverlap;
    }
    const auto at = static_cast<size_t>(kAutoStartOnEndActivationOffset - block_offset);
    // EX_JumpIfNot(0x4fa8, K2Node_Event_Activate), true call/pop, false call/pop.
    constexpr std::array<uint8_t, 44> before {
        0x07, 0xa8, 0x4f, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x02, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x45, 0x52, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x4d,
        0x45, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x4d
    };
    auto after = before;
    after[23] = 0x08; // FName index 0x0308 = StartGame; name number remains zero.
    after[24] = 0x03;
    // Require the complete condition and both branches before changing any byte.
    if (block.size() - at < before.size()) {
        return Result::NoOverlap;
    }
    const auto view = block.subspan(at, before.size());
    if (std::ranges::equal(view, after)) {
        return Result::Unchanged;
    }
    if (!std::ranges::equal(view, before)) {
        return Result::Mismatch;
    }
    std::ranges::copy(after, view.begin());
    return Result::Patched;
}

// OnAnalogueOrGestureClick calls this immediately after SetupComfortButtons
// has populated the preset button references. Select Standard synchronously
// instead of waiting for OnAnimatedOff; retain the serialized call length.
inline constexpr uint64_t kMovementStandardAfterSetupOffset = 75826518121ull;
[[nodiscard]] inline Result ApplyMovementStandardAfterSetup(
    uint64_t block_offset, std::span<uint8_t> block) {
    if (block_offset > kMovementStandardAfterSetupOffset ||
        kMovementStandardAfterSetupOffset - block_offset >= block.size()) {
        return Result::NoOverlap;
    }
    const auto at = static_cast<size_t>(kMovementStandardAfterSetupOffset - block_offset);
    constexpr std::array<uint8_t, 11> before {
        0x45, 0xc9, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x27, 0x16
    };
    constexpr std::array<uint8_t, 11> after {
        0x45, 0x88, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x0b
    };
    // The argument removal and trailing EX_Nothing are one transaction.
    if (block.size() - at < before.size()) {
        return Result::NoOverlap;
    }
    const auto view = block.subspan(at, before.size());
    if (std::ranges::equal(view, after)) {
        return Result::Unchanged;
    }
    if (!std::ranges::equal(view, before)) {
        return Result::Mismatch;
    }
    std::ranges::copy(after, view.begin());
    return Result::Patched;
}

// An exact-version alternative to attempting to fake every VR calibration
// screen: leave BP_HeadsetCalibration's ReceiveBeginPlay a valid, empty
// Kismet function. The six EX_Nothing opcodes replace EX_LocalFinalFunction's
// entry dispatch and parameters; the original Return/EndOfScript is retained.
inline constexpr uint64_t kDisableCalibrationBeginOffset = 75823936562ull;
[[nodiscard]] inline Result ApplyDisableCalibrationBegin(
    uint64_t offset, std::span<uint8_t> block) {
    if (offset > kDisableCalibrationBeginOffset ||
        kDisableCalibrationBeginOffset - offset >= block.size()) {
        return Result::NoOverlap;
    }
    const size_t at = static_cast<size_t>(kDisableCalibrationBeginOffset - offset);
    constexpr std::array<uint8_t, 9> before {
        0x1d, 0x9b, 0x33, 0x00, 0x00, 0x16, 0x04, 0x0b, 0x53
    };
    constexpr std::array<uint8_t, 9> after {
        0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x04, 0x0b, 0x53
    };
    if (block.size() - at < before.size()) {
        return Result::NoOverlap;
    }
    const auto view = block.subspan(at, before.size());
    if (std::ranges::equal(view, after)) {
        return Result::Unchanged;
    }
    if (!std::ranges::equal(view, before)) {
        return Result::Mismatch;
    }
    std::ranges::copy(after, view.begin());
    return Result::Patched;
}

// OnPanelButtonAnimatedOn contains EX_Jump to a panel-only Kismet path.
// Enter OnAnalogueSelected instead, which updates the mode and calls the
// parent-facing movement callback above. Retain the EX_Jump opcode and
// the following function call's opcode as signature anchors.
inline constexpr uint64_t kMovementChoiceJump = 75826512382ull;
[[nodiscard]] inline Result ApplyMovementChoiceJump(uint64_t block_offset,
                                                     std::span<uint8_t> block,
                                                     bool notify_parent = false) {
    if (block_offset > kMovementChoiceJump ||
        kMovementChoiceJump - block_offset >= block.size()) {
        return Result::NoOverlap;
    }
    const auto at = static_cast<size_t>(kMovementChoiceJump - block_offset);
    if (block.size() - at < 6u) {
        return Result::NoOverlap;
    }
    if (block[at] != 0x06u || block[at + 5] != 0x45u) {
        return Result::Mismatch;
    }
    constexpr std::array<uint8_t, 4> original {0x5c, 0x0d, 0x00, 0x00};
    constexpr std::array<uint8_t, 4> select_mode {0x81, 0x1a, 0x00, 0x00};
    constexpr std::array<uint8_t, 4> notify {0x51, 0x16, 0x00, 0x00};
    const auto& replacement = notify_parent ? notify : select_mode;
    bool unchanged = true;
    for (size_t i = 0; i < original.size(); ++i) {
        const auto value = block[at + 1 + i];
        if (value != original[i] && value != replacement[i]) {
            return Result::Mismatch;
        }
        unchanged &= value == replacement[i];
    }
    if (unchanged) {
        return Result::Unchanged;
    }
    std::ranges::copy(replacement, block.begin() + at + 1);
    return Result::Patched;
}

// BP_HeadsetCalibration::OnComfortPresetSelected. This optional, exact-version
// Kismet patch follows the successful comfort callback into the final-step
// setup, then redirects its jump to StartGame. The three edits form one
// transaction so a partly read APR block cannot leave broken bytecode.
inline constexpr uint64_t kAutoWizardStart = 75823918508ull;
inline constexpr uint64_t kAutoWizardEnd = 75823918657ull;

[[nodiscard]] inline Result ApplyAutoWizard(uint64_t block_offset, std::span<uint8_t> block) {
    if (block_offset > kAutoWizardStart ||
        kAutoWizardStart - block_offset >= block.size()) {
        return Result::NoOverlap;
    }
    const auto at = static_cast<size_t>(kAutoWizardStart - block_offset);
    if (block.size() - at < kAutoWizardEnd - kAutoWizardStart) {
        return Result::NoOverlap;
    }
    constexpr std::array<uint8_t, 4> original_jump {0xba, 0x0b, 0x00, 0x00};
    constexpr std::array<uint8_t, 4> replacement_jump {0xb7, 0x4f, 0x00, 0x00};
    // EX_EndFunctionParms before each EX_PopExecutionFlow, and EX_Jump
    // immediately after the second, are fixed anchors for this asset build.
    if (at == 0 || block[at - 1] != 0x16u ||
        block[at + 1] != 0x61u ||
        block[at + 142] != 0x16u ||
        block[at + 144] != 0x06u) {
        return Result::Mismatch;
    }
    const auto is_opcode = [](uint8_t v) { return v == 0x4du || v == 0x0bu; };
    if (!is_opcode(block[at]) || !is_opcode(block[at + 143])) {
        return Result::Mismatch;
    }
    bool unchanged = block[at] == 0x0bu && block[at + 143] == 0x0bu;
    for (size_t i = 0; i < 4; ++i) {
        const auto value = block[at + 145 + i];
        if (value != original_jump[i] && value != replacement_jump[i]) {
            return Result::Mismatch;
        }
        unchanged &= value == replacement_jump[i];
    }
    if (unchanged) {
        return Result::Unchanged;
    }
    block[at] = 0x0bu;
    block[at + 143] = 0x0bu;
    std::ranges::copy(replacement_jump, block.begin() + at + 145);
    return Result::Patched;
}
struct Plan {
	bool              valid = true;
	std::vector<Edit> edits;
};

// The shipped pak stores DefaultEngine.ini uncompressed at this exact
// offset. This opt-in edit changes GameDefaultMap without touching the pak.
// Trailing spaces preserve length and are trimmed by UE4's INI parser.
inline constexpr uint64_t kDefaultMapPakOffset = 61453963ull;
inline constexpr std::string_view kDefaultMapOriginal =
    "GameDefaultMap=/Game/Maps/StartupMap.StartupMap";
inline constexpr std::string_view kDefaultMapMainMenu =
    "GameDefaultMap=/Game/Maps/MainMenu.MainMenu    ";
static_assert(kDefaultMapOriginal.size() == kDefaultMapMainMenu.size());

// The next runtime-irrelevant EditorStartupMap line gives enough room to
// directly boot the cooked RiverRideV2 persistent level. Replace both full
// lines in the transient APR buffer, preserving the serialized byte length.
inline constexpr std::string_view kDefaultMapAndEditorOriginal =
    "GameDefaultMap=/Game/Maps/StartupMap.StartupMap\r\n"
    "EditorStartupMap=/Engine/Maps/Entry.Entry\r\n";
inline constexpr std::string_view kDefaultMapRiverRide =
    "GameDefaultMap=/Game/Maps/RiverRideV2/RiverRideV2_Persistent."
    "RiverRideV2_Persistent       \r\n";
static_assert(kDefaultMapAndEditorOriginal.size() == kDefaultMapRiverRide.size());

inline constexpr std::string_view kHubMapPath =
    "/Game/Maps/Hub/Hub_Persistent.Hub_Persistent";

[[nodiscard]] inline Result ApplyHubMap(uint64_t block_offset,
                                        std::span<uint8_t> block) {
    if (block_offset > kDefaultMapPakOffset ||
        kDefaultMapPakOffset - block_offset >= block.size()) {
        return Result::NoOverlap;
    }
    const auto at = static_cast<size_t>(kDefaultMapPakOffset - block_offset);
    constexpr size_t patch_size = kDefaultMapAndEditorOriginal.size();
    if (block.size() - at < patch_size) {
        return Result::NoOverlap;
    }
    std::array<char, patch_size> replacement {};
    std::fill(replacement.begin(), replacement.end() - 2, ' ');
    const std::string_view key = "GameDefaultMap=";
    if (key.size() + kHubMapPath.size() > patch_size - 2) {
        return Result::Mismatch;
    }
    std::ranges::copy(key, replacement.begin());
    std::ranges::copy(kHubMapPath, replacement.begin() + key.size());
    replacement[patch_size - 2] = '\r';
    replacement[patch_size - 1] = '\n';

    const std::string_view actual(reinterpret_cast<const char*>(block.data() + at),
                                  patch_size);
    if (actual == std::string_view(replacement.data(), replacement.size())) {
        return Result::Unchanged;
    }
    if (actual != kDefaultMapAndEditorOriginal) {
        return Result::Mismatch;
    }
    std::ranges::copy(replacement, block.begin() + at);
    return Result::Patched;
}

[[nodiscard]] inline Result ApplyRiverRideMap(uint64_t block_offset,
                                             std::span<uint8_t> block) {
    if (block_offset > kDefaultMapPakOffset ||
        kDefaultMapPakOffset - block_offset >= block.size()) {
        return Result::NoOverlap;
    }
    const auto at = static_cast<size_t>(kDefaultMapPakOffset - block_offset);
    if (block.size() - at < kDefaultMapAndEditorOriginal.size()) {
        return Result::NoOverlap;
    }
    const auto* ptr = reinterpret_cast<const char*>(block.data() + at);
    const std::string_view actual(ptr, kDefaultMapAndEditorOriginal.size());
    if (actual == kDefaultMapRiverRide) {
        return Result::Unchanged;
    }
    if (actual != kDefaultMapAndEditorOriginal) {
        return Result::Mismatch;
    }
    std::ranges::copy(kDefaultMapRiverRide, block.begin() + at);
    return Result::Patched;
}

[[nodiscard]] inline Result ApplyMainMenuMap(uint64_t block_offset,
                                             std::span<uint8_t> block) {
    if (block_offset > kDefaultMapPakOffset ||
        kDefaultMapPakOffset - block_offset > block.size()) {
        return Result::NoOverlap;
    }
    const auto at = static_cast<size_t>(kDefaultMapPakOffset - block_offset);
    if (block.size() - at < kDefaultMapOriginal.size()) {
        return Result::NoOverlap;
    }
    const auto* ptr = reinterpret_cast<const char*>(block.data() + at);
    const std::string_view actual(ptr, kDefaultMapOriginal.size());
    if (actual == kDefaultMapMainMenu) {
        return Result::Unchanged;
    }
    if (actual != kDefaultMapOriginal) {
        return Result::Mismatch;
    }
    std::ranges::copy(kDefaultMapMainMenu, block.begin() + at);
    return Result::Patched;
}

[[nodiscard]] inline std::string_view Trim(std::string_view s) {
	const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
	while (!s.empty() && is_space(s.front())) {
		s.remove_prefix(1);
	}
	while (!s.empty() && is_space(s.back())) {
		s.remove_suffix(1);
	}
	return s;
}

[[nodiscard]] inline const Event* FindEvent(std::string_view name) {
	const auto it = std::ranges::find_if(kEvents, [name](const Event& event) {
		return event.name == name;
	});
	return it == kEvents.end() ? nullptr : &*it;
}

[[nodiscard]] inline bool ParseEntry(std::string_view value, uint32_t* entry) {
	value = Trim(value);
	if (const auto* event = FindEvent(value); event != nullptr) {
		*entry = event->graph_entry;
		return true;
	}
	int base = 10;
	if (value.starts_with("0x") || value.starts_with("0X")) {
		value.remove_prefix(2);
		base = 16;
	}
	if (value.empty()) {
		return false;
	}
	uint32_t parsed = 0;
	const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(),
	                                       parsed, base);
	if (ec != std::errc {} || end != value.data() + value.size() || parsed >= 0x8000u) {
		return false;
	}
	*entry = parsed;
	return true;
}

[[nodiscard]] inline std::array<uint8_t, 4> EntryBytes(uint32_t entry) {
	return {static_cast<uint8_t>(entry), static_cast<uint8_t>(entry >> 8u),
	        static_cast<uint8_t>(entry >> 16u), static_cast<uint8_t>(entry >> 24u)};
}

// Examples: "seated", "begin=startgame", "intro=seated,fadein=startgame".
// A bare destination is shorthand for "begin=<destination>".
// Invalid/duplicate entries reject the entire plan rather than applying part.
[[nodiscard]] inline Plan ParsePlan(std::string_view spec) {
	Plan plan {};
	if (Trim(spec).empty()) {
		return plan;
	}
	while (!spec.empty()) {
		const auto comma = spec.find(',');
		auto       item  = Trim(spec.substr(0, comma));
		spec = comma == std::string_view::npos ? std::string_view {} : spec.substr(comma + 1);
		const auto equals = item.find('=');
		const auto source = Trim(equals == std::string_view::npos ? "begin" : item.substr(0, equals));
		const auto target = Trim(equals == std::string_view::npos ? item : item.substr(equals + 1));
		const auto* event = FindEvent(source);
		uint32_t target_entry = 0;
		if (event == nullptr || !ParseEntry(target, &target_entry) ||
		    std::ranges::any_of(plan.edits, [source](const Edit& edit) {
			    return edit.event == source;
		    })) {
			return {false, {}};
		}
		plan.edits.push_back({event->name,
		                      kUexpBase + event->serialized_offset - kUassetLength + kEntryOffset,
		                      EntryBytes(event->graph_entry), EntryBytes(target_entry)});
		// A trailing comma is invalid rather than silently ignored.
		if (comma != std::string_view::npos && spec.empty()) {
			return {false, {}};
		}
	}
	return plan;
}

// Checks the signature of the compiled UFunction call when those bytes are
// present: EX_IntConst (0x1d), 4-byte graph offset, then EX_EndFunctionParms,
// EX_Return, EX_Nothing, EX_EndOfScript (16 04 0b 53).
[[nodiscard]] inline Result Apply(const Edit& edit, uint64_t block_offset,
                                  std::span<uint8_t> block) {
    if (block.empty() || block_offset >= edit.file_offset + 4u ||
        (block_offset < edit.file_offset &&
         block.size() <= edit.file_offset - block_offset)) {
        return Result::NoOverlap;
    }
    auto check = [&](uint64_t addr, uint8_t expected) {
        if (addr < block_offset || addr - block_offset >= block.size()) {
            return true;
        }
        return block[static_cast<size_t>(addr - block_offset)] == expected;
    };
    constexpr std::array<uint8_t, 5> kSuffix {0x16u, 0x04u, 0x0bu, 0x53u, 0x00u};
    if (!check(edit.file_offset - 1u, 0x1du)) {
        return Result::Mismatch;
    }
    for (uint64_t i = 0; i < kSuffix.size(); ++i) {
        if (!check(edit.file_offset + 4u + i, kSuffix[i])) {
            return Result::Mismatch;
        }
    }

    bool needs_patch = false;
    for (uint64_t i = 0; i < 4u; ++i) {
        const auto addr = edit.file_offset + i;
        if (addr < block_offset || addr - block_offset >= block.size()) {
            continue;
        }
        const auto byte = block[static_cast<size_t>(addr - block_offset)];
        if (byte != edit.original[i] && byte != edit.replacement[i]) {
            return Result::Mismatch;
        }
        needs_patch = needs_patch || byte != edit.replacement[i];
    }
    if (!needs_patch) {
        return Result::Unchanged;
    }
    for (uint64_t i = 0; i < 4u; ++i) {
        const auto addr = edit.file_offset + i;
        if (addr >= block_offset && addr - block_offset < block.size()) {
            block[static_cast<size_t>(addr - block_offset)] = edit.replacement[i];
        }
    }
    return Result::Patched;
}
} // namespace Libs::HorizonCalibrationPatch
