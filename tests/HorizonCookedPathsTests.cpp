#include "kernel/horizonCookedPaths.h"

#include <cstdlib>
#include <cstdio>
#include <string_view>

using Libs::LibKernel::FileSystem::ResolveHorizonCookedMoviePath;

static void Expect(std::string_view input, std::string_view expected) {
	const auto result = ResolveHorizonCookedMoviePath(input);
	if (!result || *result != expected) {
		std::fprintf(stderr, "bad cooked path mapping: %.*s\n",
		             static_cast<int>(input.size()), input.data());
		std::abort();
	}
}

static void Reject(std::string_view input) {
	if (ResolveHorizonCookedMoviePath(input).has_value()) {
		std::fprintf(stderr, "unexpected cooked path mapping: %.*s\n",
		             static_cast<int>(input.size()), input.data());
		std::abort();
	}
}

int main() {
	Expect("/app0/d:/work/projects/azi/content/movies/tutorials/shooting_tutorial_character.bk2",
	       "/app0/azimuth/content/movies/tutorials/shooting_tutorial_character.bk2");
	Expect("/app0/azimuth/content/d:/work/projects/azi/content/movies/tutorials/shooting_tutorial_character.bk2",
	       "/app0/azimuth/content/movies/tutorials/shooting_tutorial_character.bk2");
	Expect("/app0/d:/work/projects/azi/content/movies/rotation_movement_tutorial.bk2",
	       "/app0/azimuth/content/movies/rotation_movement_tutorial.bk2");
	Reject("/app0/azimuth/content/movies/tutorials/shooting_tutorial_character.bk2");
	Reject("/app0/d:/work/projects/azi/content/movies/tutorials/../other.bk2");
	Reject("/app0/d:/work/projects/azi/content/movies/../private.bk2");
	Reject("/app0/d:/work/projects/azi/content/movies/foo:bar.bk2");
	Reject("/app0/d:/work/projects/azi/content/movies/foo\\bar.bk2");
	Reject("/app0/d:/work/projects/azi/content/movies/tutorials/");
	Reject("/app0/d:/work/projects/azi/content/movies/tutorials/not_a_movie.uasset");
	Reject("/app1/d:/work/projects/azi/content/movies/foo.bk2");
	std::puts("HorizonCookedPathsTests passed");
}
