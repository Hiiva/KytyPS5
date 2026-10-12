#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace Libs::LibKernel::FileSystem {

// Horizon's UE4 media assets contain editor-machine paths such as
// D:/work/projects/azi/content/movies/tutorials/foo.bk2. On PS5 these are
// under /app0/azimuth/content/movies; on Windows the embedded drive letter
// is not a legal relative filename. Only rewrite this exact known prefix.
[[nodiscard]] inline std::optional<std::string> ResolveHorizonCookedMoviePath(
    std::string_view mounted_name) {
	constexpr std::string_view app_root = "/app0/";
	constexpr std::string_view content_root = "azimuth/content/";
	constexpr std::string_view cook_root = "d:/work/projects/azi/content/movies/";
	if (!mounted_name.starts_with(app_root)) {
		return std::nullopt;
	}
	mounted_name.remove_prefix(app_root.size());
	if (mounted_name.starts_with(content_root)) {
		mounted_name.remove_prefix(content_root.size());
	}
	if (!mounted_name.starts_with(cook_root)) {
		return std::nullopt;
	}
	mounted_name.remove_prefix(cook_root.size());
	if (mounted_name.empty() || !mounted_name.ends_with(".bk2")) {
		return std::nullopt;
	}
	for (auto remaining = mounted_name; !remaining.empty();) {
		const auto separator = remaining.find('/');
		const auto component = remaining.substr(0, separator);
		if (component.empty() || component == "." || component == ".." ||
		    component.find_first_of("<>:\"\\|?*") != std::string_view::npos) {
			return std::nullopt;
		}
		if (separator == std::string_view::npos) {
			break;
		}
		remaining.remove_prefix(separator + 1u);
	}
	std::string result = "/app0/azimuth/content/movies/";
	result += mounted_name;
	return result;
}

} // namespace Libs::LibKernel::FileSystem
