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

inline constexpr std::array<Event, 23> kEvents {{
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
}};

struct Edit {
	std::string_view     event;
	uint64_t             file_offset;
	std::array<uint8_t, 4> original;
	std::array<uint8_t, 4> replacement;
};

struct Plan {
	bool              valid = true;
	std::vector<Edit> edits;
};

enum class Result { NoOverlap, Unchanged, Patched, Mismatch };

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
