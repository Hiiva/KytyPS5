// Exercise the production controller with deterministic host output and time.
#include <SDL3/SDL.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "ControllerSettingsTests: %s\n", text);
		std::abort();
	}
}

struct Rumble {
	Uint16 large, small;
	Uint32 duration;
};
Uint64                             now = 1000;
std::vector<Rumble>                rumble;
std::vector<Rumble>                haptics;
std::vector<std::array<Uint8, 32>> effects;
bool                               haptics_handles_rumble = false;
std::array<Uint8, 0x68>             vr_read_buffer {};
bool                               relative_mouse_enabled = false;
bool                               relative_mouse_allowed = true;
bool                               window_has_focus = true;
float                              mouse_dx = 0.0f, mouse_dy = 0.0f;
SDL_Window* const                  test_window = reinterpret_cast<SDL_Window*>(uintptr_t {1});
} // namespace

namespace Fake {
Uint64 GetTicks() {
	return now;
}
SDL_Gamepad* GetGamepadFromID(SDL_JoystickID id) {
	return id == 1 || id == 2 ? reinterpret_cast<SDL_Gamepad*>(static_cast<uintptr_t>(id))
	                          : nullptr;
}
SDL_GamepadType GetGamepadType(SDL_Gamepad*) {
	return SDL_GAMEPAD_TYPE_PS5;
}
bool RumbleGamepad(SDL_Gamepad*, Uint16 large, Uint16 small, Uint32 duration) {
	rumble.push_back({large, small, duration});
	return true;
}
bool SendGamepadEffect(SDL_Gamepad*, const void* data, int size) {
	Check(size == 32, "unexpected DualSense effect size");
	std::array<Uint8, 32> effect {};
	std::memcpy(effect.data(), data, effect.size());
	effects.push_back(effect);
	return true;
}
bool SetGamepadLED(SDL_Gamepad*, Uint8, Uint8, Uint8) {
	return true;
}
bool GamepadHasSensor(SDL_Gamepad*, SDL_SensorType) {
	return false;
}
bool SetGamepadSensorEnabled(SDL_Gamepad*, SDL_SensorType, bool) {
	return true;
}
void CloseGamepad(SDL_Gamepad*) {}
void Delay(Uint32) {}
bool SetWindowRelativeMouseMode(SDL_Window*, bool enabled) {
	if (!relative_mouse_allowed) { return false; }
	relative_mouse_enabled = enabled;
	return true;
}
SDL_MouseButtonFlags GetRelativeMouseState(float* x, float* y) {
	if (x != nullptr) { *x = mouse_dx; }
	if (y != nullptr) { *y = mouse_dy; }
	mouse_dx = mouse_dy = 0.0f;
	return 0;
}
SDL_Window* GetKeyboardFocus() { return window_has_focus ? test_window : nullptr; }
bool WaitEventTimeout(SDL_Event*, Sint32) { return false; }
} // namespace Fake

#define SDL_GetTicks                Fake::GetTicks
#define SDL_GetGamepadFromID        Fake::GetGamepadFromID
#define SDL_GetGamepadType          Fake::GetGamepadType
#define SDL_RumbleGamepad           Fake::RumbleGamepad
#define SDL_SendGamepadEffect       Fake::SendGamepadEffect
#define SDL_SetGamepadLED           Fake::SetGamepadLED
#define SDL_GamepadHasSensor        Fake::GamepadHasSensor
#define SDL_SetGamepadSensorEnabled Fake::SetGamepadSensorEnabled
#define SDL_CloseGamepad            Fake::CloseGamepad
#define SDL_Delay                   Fake::Delay
#include "libs/controller.cpp"
#undef SDL_GetTicks
#undef SDL_GetGamepadFromID
#undef SDL_GetGamepadType
#undef SDL_RumbleGamepad
#undef SDL_SendGamepadEffect
#undef SDL_SetGamepadLED
#undef SDL_GamepadHasSensor
#undef SDL_SetGamepadSensorEnabled
#undef SDL_CloseGamepad
#undef SDL_Delay

// Exercise the real mouse modes/keymap routing without opening a host window.
#define SDL_GetTicks                   Fake::GetTicks
#define SDL_SetWindowRelativeMouseMode Fake::SetWindowRelativeMouseMode
#define SDL_GetRelativeMouseState      Fake::GetRelativeMouseState
#define SDL_GetKeyboardFocus           Fake::GetKeyboardFocus
#define SDL_WaitEventTimeout           Fake::WaitEventTimeout
#include "graphics/presentation/window/hostInput.cpp"
#undef SDL_GetTicks
#undef SDL_SetWindowRelativeMouseMode
#undef SDL_GetRelativeMouseState
#undef SDL_GetKeyboardFocus
#undef SDL_WaitEventTimeout

namespace Libs::Controller::DualSenseHaptics {
bool SetVibration(int, uint8_t large_motor, uint8_t small_motor, uint32_t duration_ms) {
	haptics.push_back({large_motor, small_motor, duration_ms});
	return haptics_handles_rumble;
}
void Shutdown() {}
} // namespace Libs::Controller::DualSenseHaptics

namespace Libs::LibKernel {
uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	return now * 1000;
}
} // namespace Libs::LibKernel

namespace Libs::Graphics {
bool HostMemoryRangeIsReadable(uint64_t addr, uint64_t size) {
	return addr == reinterpret_cast<uint64_t>(vr_read_buffer.data()) &&
	       size == vr_read_buffer.size();
}
} // namespace Libs::Graphics

namespace Loader::Timer {
double GetTimeMs() {
	return static_cast<double>(now);
}
} // namespace Loader::Timer

namespace {
using namespace Libs::Controller;

struct Controller {
	Controller() {
		now                    = 1000;
		haptics_handles_rumble = false;
		Initialize();
		Check(PadInit() == 0 && PadOpen(Config::GetUserId(), 0, 0, nullptr) == 1,
		      "the standard controller port did not open");
		Connect(1);
		Check(GetSettingScale(Setting::SpeakerVolume) ==
		              Config::GetControllerSpeakerVolume() / 50.0f &&
		          GetSettingScale(Setting::VibrationIntensity) ==
		              Config::GetControllerVibrationIntensity() / 100.0f &&
		          GetSettingScale(Setting::TriggerEffectIntensity) == 1.0f,
		      "controller initialization retained old settings");
		rumble.clear();
		haptics.clear();
		effects.clear();
	}
	~Controller() { Shutdown(); }
};

class ScopedEnvironment {
public:
	ScopedEnvironment(const char* name, const char* value): m_name(name) {
		if (const char* previous = std::getenv(name)) {
			m_present = true;
			m_previous = previous;
		}
		Set(value);
	}
	~ScopedEnvironment() { Set(m_present ? m_previous.c_str() : nullptr); }
private:
	void Set(const char* value) {
#ifdef _WIN32
		Check(_putenv_s(m_name, value != nullptr ? value : "") == 0, "environment update failed");
#else
		Check((value != nullptr ? setenv(m_name, value, 1) : unsetenv(m_name)) == 0,
		      "environment update failed");
#endif
	}
	const char* m_name;
	bool m_present = false;
	std::string m_previous;
};

struct VirtualPose {
	std::array<float, 3> position;
	std::array<float, 4> orientation;
};

VirtualPose ReadHand(int side) {
	VirtualPose pose;
	VirtualHandGetPose(side, pose.position.data(), pose.orientation.data());
	return pose;
}

template <size_t N>
bool Near(const std::array<float, N>& a, const std::array<float, N>& b) {
	for (size_t axis = 0; axis < N; ++axis) {
		if (!std::isfinite(a[axis]) || std::abs(a[axis] - b[axis]) > 0.00001f) { return false; }
	}
	return true;
}

bool SamePose(const VirtualPose& a, const VirtualPose& b) {
	return Near(a.position, b.position) && Near(a.orientation, b.orientation);
}

bool InHandWorkspace(const VirtualPose& pose) {
	const float head[3] = {0.0f, 1.6f, 0.0f};
	for (size_t axis = 0; axis < pose.position.size(); ++axis) {
		if (!std::isfinite(pose.position[axis]) || std::abs(pose.position[axis] - head[axis]) > 2.00001f) {
			return false;
		}
	}
	float norm = 0.0f;
	for (float component: pose.orientation) { norm += component * component; }
	return std::isfinite(norm) && std::abs(norm - 1.0f) < 0.00001f;
}

void TestIndependentHandPositions() {
	Config::ConfigOptions options;
	options.vr_enabled = true;
	Config::Load(options);
	Controller controller;
	ResetInputState();
	VirtualHeadResetPose();
	const auto neutral_left = ReadHand(0), neutral_right = ReadHand(1);
	Check(Near(neutral_left.position, {-0.25f, 1.26f, -0.45f}) &&
	          Near(neutral_right.position, {0.25f, 1.26f, -0.45f}), "neutral hand positions changed");

	VirtualHeadMouseMotion(180.0f, -100.0f);
	VirtualHandMouseMotion(0, 35.0f, -20.0f);
	const auto before_left = ReadHand(0), before_right = ReadHand(1);
	VirtualHandBeginPositionControl(0);
	VirtualHandBeginPositionControl(1);
	Check(SamePose(ReadHand(0), before_left) && SamePose(ReadHand(1), before_right),
	      "entering position control changed the current position or wrist orientation");
	VirtualHeadMouseMotion(-310.0f, 170.0f);
	VirtualHandBeginPositionControl(0);
	Check(SamePose(ReadHand(0), before_left) && SamePose(ReadHand(1), before_right),
	      "head motion or re-entering position control dragged a detached hand");
	VirtualHandMouseMotion(0, -45.0f, 30.0f);
	Check(Near(ReadHand(0).position, before_left.position) &&
	          !Near(ReadHand(0).orientation, before_left.orientation) &&
	          SamePose(ReadHand(1), before_right), "left wrist aim moved a hand or changed the right wrist");
	{
		const ScopedEnvironment left_override("KYTY_VR_LEFT_AIM_YAW", "0.8");
		const ScopedEnvironment right_override("KYTY_VR_RIGHT_AIM_PITCH", "-0.6");
		VirtualHeadResetPose();
		VirtualHeadMouseMotion(-170.0f, 90.0f);
		const auto overridden_left = ReadHand(0), overridden_right = ReadHand(1);
		VirtualHandBeginPositionControl(0);
		VirtualHandBeginPositionControl(1);
		VirtualHeadMouseMotion(210.0f, -130.0f);
		Check(SamePose(ReadHand(0), overridden_left) && SamePose(ReadHand(1), overridden_right),
		      "position control discarded a nondefault wrist aim override");
	}

	VirtualHeadResetPose();
	VirtualHandBeginPositionControl(0);
	VirtualHandBeginPositionControl(1);
	constexpr float quarter_turn_pixels = -1.57079632679f / 0.0025f;
	VirtualHeadMouseMotion(quarter_turn_pixels, 0.0f);
	VirtualHandMovePosition(1, 0.1f, 0.2f, 0.3f);
	Check(Near(ReadHand(1).position, {-0.05f, 1.46f, -0.55f}) &&
	          SamePose(ReadHand(0), neutral_left),
	      "position deltas did not use the current view's right/up/forward axes");
	VirtualHandAdjustReach(1, 1.0f);
	Check(Near(ReadHand(1).position, {-0.13f, 1.46f, -0.55f}),
	      "independent reach did not move along the current view depth");
	const auto stationary_right = ReadHand(1);
	SetAxis(1, Axis::RightX, 255);
	for (int frame = 0; frame < 10; ++frame) { now += 33; (void)ReadHand(1); }
	ResetInputState();
	Check(SamePose(ReadHand(1), stationary_right), "right-stick head turning dragged an independent hand");

	VirtualHeadResetPose();
	const float meeting_point[3] = {0.1f, 2.3f, 0.3f};
	VirtualHandSetPosition(0, meeting_point);
	VirtualHandSetPosition(1, meeting_point);
	Check(Near(ReadHand(0).position, {0.1f, 2.3f, 0.3f}) &&
	          Near(ReadHand(1).position, ReadHand(0).position),
	      "hands could not meet above and behind the headset");
	VirtualHandMovePosition(0, -0.4f, 0.0f, 0.0f);
	Check(Near(ReadHand(0).position, {-0.3f, 2.3f, 0.3f}) &&
	          Near(ReadHand(1).position, {0.1f, 2.3f, 0.3f}), "moving the left hand changed the right position");
	const auto valid_left = ReadHand(0);
	const float invalid_position[3] = {0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f};
	VirtualHandSetPosition(0, invalid_position);
	VirtualHandSetPosition(0, nullptr);
	VirtualHandMovePosition(0, 0.0f, std::numeric_limits<float>::infinity(), 0.0f);
	VirtualHandBeginPositionControl(-1);
	VirtualHandMovePosition(2, 1.0f, 1.0f, 1.0f);
	VirtualHandSetPosition(2, meeting_point);
	Check(SamePose(ReadHand(0), valid_left), "invalid hand input corrupted a valid pose");
	const float outside_workspace[3] = {-100.0f, 100.0f, 100.0f};
	VirtualHandSetPosition(0, outside_workspace);
	Check(Near(ReadHand(0).position, {-2.0f, 3.6f, 2.0f}), "absolute position escaped the desktop workspace");
	VirtualHandMovePosition(0, 100.0f, -100.0f, 100.0f);
	Check(Near(ReadHand(0).position, {2.0f, -0.4f, -2.0f}), "position motion escaped the desktop workspace");

	VirtualHeadResetPose();
	Check(SamePose(ReadHand(0), neutral_left) && SamePose(ReadHand(1), neutral_right),
	      "desktop reset did not restore both neutral hands");
	VirtualHandAdjustReach(1, 100.0f);
	Check(Near(ReadHand(1).position, {0.25f, 1.26f, -1.2f}), "legacy maximum reach changed after reset");
	VirtualHandAdjustReach(1, -100.0f);
	Check(Near(ReadHand(1).position, {0.25f, 1.26f, -0.2f}), "legacy minimum reach changed after reset");
	VirtualHeadMouseMotion(100.0f, 0.0f);
	Check(!Near(ReadHand(0).position, neutral_left.position), "reset left an untouched hand detached");
	VirtualHeadResetPose();
}

void PressHostKey(SDL_Keycode key) {
	Libs::Graphics::HostInputKey(key, true);
	Libs::Graphics::HostInputKey(key, false);
}

void PollHostMouse(float dx, float dy) {
	SDL_Event event {};
	// A newly selected mode first drains deltas from its predecessor.
	(void)Libs::Graphics::HostInputWaitEvent(&event, 0);
	now += 33;
	mouse_dx = dx;
	mouse_dy = dy;
	(void)Libs::Graphics::HostInputWaitEvent(&event, 0);
}

void TestDesktopHandBindings(bool custom) {
	Config::ConfigOptions options;
	options.vr_enabled = true;
	if (custom) { options.keymap = {"Square=F2", "Cross=F3", "Circle=F5"}; }
	Config::Load(options);
	Controller controller;
	ResetInputState();
	VirtualHeadResetPose();
	using namespace Libs::Graphics;
	HostInputInit(test_window);
	if (custom) {
		const std::pair<SDL_Keycode, uint32_t> keys[] = {
		    {SDLK_F2, PAD_BUTTON_SQUARE}, {SDLK_F3, PAD_BUTTON_CROSS}, {SDLK_F5, PAD_BUTTON_CIRCLE}};
		for (const auto [key, button]: keys) {
			HostInputKey(key, true);
			PadData data {};
			Check(PadReadState(1, &data) == OK && data.buttons == button && !relative_mouse_enabled,
			      "a desktop hand hotkey overrode its custom binding");
			HostInputKey(key, false);
			Check(PadReadState(1, &data) == OK && data.buttons == 0, "custom hotkey release was lost");
		}
		HostInputShutdown();
		return;
	}

	const auto left = ReadHand(0), right = ReadHand(1);
	relative_mouse_allowed = false;
	PressHostKey(SDLK_F2);
	Check(!relative_mouse_enabled, "failed relative-mouse capture enabled position control");
	VirtualHeadMouseMotion(30.0f, 0.0f);
	Check(!SamePose(ReadHand(0), left), "failed mouse capture detached the hand");
	VirtualHeadResetPose();
	relative_mouse_allowed = true;
	PressHostKey(SDLK_F2);
	Check(relative_mouse_enabled, "F2 did not capture left-hand position input");
	HostInputMouseButton(SDL_BUTTON_MIDDLE, true);
	PollHostMouse(100.0f, -50.0f);
	Check(Near(ReadHand(0).position, {-0.05f, 1.36f, -0.45f}) && SamePose(ReadHand(1), right),
	      "F2 mouse motion did not move only the left hand at two millimetres per pixel");
	HostInputMouseWheel(1.0f, false);
	Check(Near(ReadHand(0).position, {-0.05f, 1.36f, -0.53f}) && SamePose(ReadHand(1), right),
	      "unmodified wheel did not select the left position-control hand");
	const auto placed_left = ReadHand(0);
	PressHostKey(SDLK_F3);
	HostInputMouseButton(SDL_BUTTON_LEFT, true);
	PollHostMouse(-50.0f, 25.0f);
	Check(Near(ReadHand(1).position, {0.15f, 1.21f, -0.45f}) && SamePose(ReadHand(0), placed_left),
	      "F3 failed to preserve the left hand while positioning the right");
	PadData data {};
	Check(PadReadState(1, &data) == OK && data.analog_buttons_l2 == 255 &&
	          data.analog_buttons_r2 == 255 && (data.buttons & PAD_BUTTON_L3) == 0,
	      "hand placement lost held grips or emitted a Shift/L3 press");
	HostInputMouseButton(SDL_BUTTON_MIDDLE, false);
	Check(PadReadState(1, &data) == OK && data.analog_buttons_l2 == 0 && data.analog_buttons_r2 == 255,
	      "releasing the left grip also released the right");
	HostInputMouseButton(SDL_BUTTON_LEFT, false);
	HostInputMouseWheel(1.0f, false);
	Check(Near(ReadHand(1).position, {0.15f, 1.21f, -0.53f}), "right-hand wheel routing was lost");
	HostInputMouseWheel(1.0f, true);
	Check(Near(ReadHand(0).position, {-0.05f, 1.36f, -0.61f}), "legacy Shift/wheel left selection changed");
	const auto before_aim = ReadHand(0), parked_right = ReadHand(1);
	PressHostKey(SDLK_F5);
	PollHostMouse(50.0f, -25.0f);
	Check(Near(ReadHand(0).position, before_aim.position) &&
	          !Near(ReadHand(0).orientation, before_aim.orientation) && SamePose(ReadHand(1), parked_right),
	      "F5 did not switch from right position to independent left wrist aim");
	const auto parked_left = ReadHand(0);
	HostInputToggleMouseToHead();
	PollHostMouse(120.0f, -60.0f);
	Check(SamePose(ReadHand(0), parked_left) && SamePose(ReadHand(1), parked_right),
	      "mouse head-look dragged placed hands");
	HostInputToggleMouseToHand(1);
	PollHostMouse(-40.0f, 30.0f);
	Check(Near(ReadHand(1).position, parked_right.position) &&
	          !Near(ReadHand(1).orientation, parked_right.orientation) && SamePose(ReadHand(0), parked_left),
	      "legacy right wrist aim retained another mouse mode");
	PressHostKey(SDLK_F3);
	const auto before_focus_loss = ReadHand(1);
	window_has_focus = false;
	PollHostMouse(200.0f, 100.0f);
	window_has_focus = true;
	PollHostMouse(0.0f, 0.0f);
	Check(SamePose(ReadHand(1), before_focus_loss), "focus loss/re-entry applied stale hand motion");
	HostInputToggleMouseToJoystick();
	PollHostMouse(20.0f, 0.0f);
	Check(PadReadState(1, &data) == OK && data.right_stick_x > 128,
	      "F7 mode retained hand positioning instead of emitting stick input");
	PressHostKey(SDLK_F2);
	Check(PadReadState(1, &data) == OK && data.right_stick_x == 128 && data.right_stick_y == 128,
	      "entering hand positioning retained mouse-generated stick deflection");
	VirtualHeadResetPose();
	Check(SamePose(ReadHand(0), left) && SamePose(ReadHand(1), right), "F9 in position mode did not recenter");
	HostInputMouseWheel(1.0f, false);
	const auto after_reset_wheel = ReadHand(0);
	Check(Near(after_reset_wheel.position, {-0.25f, 1.26f, -0.53f}),
	      "wheel after F9 did not resume the selected hand's depth control");
	VirtualHeadMouseMotion(-80.0f, -20.0f);
	Check(SamePose(ReadHand(0), after_reset_wheel), "wheel after F9 left the hand following head turns");
	VirtualHeadResetPose();
	PollHostMouse(50.0f, -25.0f);
	Check(Near(ReadHand(0).position, {-0.15f, 1.31f, -0.45f}) && SamePose(ReadHand(1), right),
	      "XY motion after F9 did not resume the selected hand from neutral");
	VirtualHeadMouseMotion(190.0f, -110.0f);
	const float extreme = std::numeric_limits<float>::max();
	PollHostMouse(extreme, -extreme);
	HostInputMouseWheel(extreme, false);
	Check(InHandWorkspace(ReadHand(0)), "finite extreme mouse/wheel input produced an invalid pose");
	VirtualHandMouseMotion(0, extreme, -extreme);
	VirtualHandMovePosition(0, extreme, -extreme, extreme);
	Check(InHandWorkspace(ReadHand(0)), "finite extreme wrist/direct motion produced an invalid pose");
	const auto retained = ReadHand(0);
	PressHostKey(SDLK_F2);
	Check(!relative_mouse_enabled && SamePose(ReadHand(0), retained), "leaving position mode reset the hand");
	VirtualHeadResetPose();
	Check(SamePose(ReadHand(0), left) && SamePose(ReadHand(1), right), "reset after mouse control lost neutral pose");
	HostInputShutdown();
	options.vr_enabled = false;
	Config::Load(options);
	PressHostKey(SDLK_F2);
	Check(!relative_mouse_enabled, "hand positioning captured the mouse outside VR mode");
}

DualSenseEffects LastEffect() {
	Check(!effects.empty(), "missing trigger output");
	DualSenseEffects effect {};
	std::memcpy(&effect, effects.back().data(), sizeof(effect));
	return effect;
}

int ZoneStrength(const uint8_t* effect, int zone) {
	const unsigned active = effect[1] | (effect[2] << 8u);
	const uint32_t packed = effect[3] | (effect[4] << 8u) | (effect[5] << 16u) |
	                        (static_cast<uint32_t>(effect[6]) << 24u);
	return (active & (1u << zone)) != 0 ? 1 + ((packed >> (3 * zone)) & 7u) : 0;
}

void SetRumble(uint8_t large, uint8_t small) {
	const PadVibrationParam param {large, small};
	Check(PadSetVibration(1, &param) == 0, "vibration request failed");
}

void TestPadPortOwnership() {
	Initialize();
	const int user = Config::GetUserId();
	constexpr auto not_initialized = static_cast<int32_t>(0x80920005u);
	constexpr auto already_opened = static_cast<int32_t>(0x80920004u);
	constexpr auto invalid_handle = static_cast<int32_t>(0x80920003u);
	constexpr auto no_handle = static_cast<int32_t>(0x80920008u);
	PadData data {};
	Check(PadOpen(user, 0, 0, nullptr) == not_initialized &&
	          PadGetHandle(user, 0, 0) == not_initialized &&
	          PadReadState(1, &data) == not_initialized,
	      "an uninitialized library exposed a controller port");
	Check(PadInit() == 0 && PadGetHandle(user, 0, 0) == no_handle,
	      "initializing the library opened a controller port");
	const int handle = PadOpen(user, 0, 0, nullptr);
	Check(handle == 1 && PadInit() == 0 && PadGetHandle(user, 0, 0) == handle,
	      "reinitializing the library lost the open port");
	for (int duplicate = 0; duplicate < 4; duplicate++) {
		const int rejected = PadOpen(user, 0, 0, nullptr);
		Check(rejected == already_opened,
		      "opening one pad five times exposed duplicate connected controllers");
		Check(PadClose(rejected) == invalid_handle && PadGetHandle(user, 0, 0) == handle,
		      "closing a rejected duplicate released the original owner");
	}
	Check(PadGetHandle(user + 1, 0, 0) == no_handle &&
	          PadGetHandle(user, 2, 0) == no_handle && PadGetHandle(user, 0, 1) == no_handle,
	      "a different port tuple aliased the open standard pad");
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, true);
	Check(PadReadState(handle, &data) == 0 && data.connected &&
	          data.buttons == PAD_BUTTON_CROSS,
	      "keyboard Cross did not reach the owned port");
	SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, false);
	Connect(1);
	SetButton(1, PAD_BUTTON_CROSS, true);
	Check(PadReadState(handle, &data) == 0 && data.buttons == PAD_BUTTON_CROSS,
	      "the physical controller did not share the owned port");
	Disconnect(1);
	Check(PadGetHandle(user, 0, 0) == handle && PadOpen(user, 0, 0, nullptr) == already_opened,
	      "host disconnection released guest port ownership");
	Check(PadClose(handle) == 0 && PadGetHandle(user, 0, 0) == no_handle &&
	          PadReadState(handle, &data) == invalid_handle &&
	          PadRead(handle, &data, 1) == invalid_handle && PadClose(handle) == invalid_handle,
	      "a closed port retained a usable handle");
	Check(PadOpen(user, 0, 0, nullptr) == handle && PadReadState(handle, &data) == 0,
	      "a closed port could not reopen");
	Shutdown();
}

struct SenseSample {
	uint32_t buttons;
	uint8_t x, y;
	float left_trigger, right_trigger;
};

SenseSample ReadSense(int handle) {
	vr_read_buffer.fill(0xa5);
	Check(PadVrControllerReadState(handle, vr_read_buffer.data()) == OK,
	      "virtual Sense state read failed");
	Check((vr_read_buffer[0x40] & 1u) != 0, "virtual Sense did not report connected");
	uint32_t buttons = 0;
	std::memcpy(&buttons, vr_read_buffer.data(), sizeof(buttons));
	// Horizon's native input consumer (0x90110819f, 0x9011082fa/306) decodes
	// sticks and named left/right trigger events from this independent layout.
	return {buttons, vr_read_buffer[4], vr_read_buffer[5],
	        vr_read_buffer[8] / 255.0f, vr_read_buffer[9] / 255.0f};
}

void TestVirtualSenseInput() {
	Config::ConfigOptions options;
	options.vr_enabled = true;
	Config::Load(options);
	{
		Controller controller;
		ResetInputState();
		const int left = PadOpen(Config::GetUserId(), 3, 0, nullptr);
		const int right = PadOpen(Config::GetUserId(), 4, 0, nullptr);
		Check(left > 0 && right > 0 && left != right, "virtual Sense ports did not open independently");
		auto check_triggers = [&](float expected_left, float expected_right) {
			const auto l = ReadSense(left), r = ReadSense(right);
			Check(std::fabs(l.left_trigger - expected_left) < 0.000001f &&
			          l.right_trigger == 0.0f && r.left_trigger == 0.0f &&
			          std::fabs(r.right_trigger - expected_right) < 0.000001f,
			      "native Sense trigger events lost input or crossed hands");
			Check(((l.buttons & 0x100u) != 0) == (expected_left > 0.0f) &&
			          (l.buttons & 0x200u) == 0 && (r.buttons & 0x100u) == 0 &&
			          ((r.buttons & 0x200u) != 0) == (expected_right > 0.0f),
			      "Sense digital trigger states disagree with the analog events");
		};
		check_triggers(0.0f, 0.0f);
		SetAxis(1, Axis::LeftX, 17);
		SetAxis(1, Axis::LeftY, 34);
		SetAxis(1, Axis::RightX, 201);
		SetAxis(1, Axis::RightY, 219);
		const auto l = ReadSense(left), r = ReadSense(right);
		Check(l.x == 17 && l.y == 34 && r.x == 201 && r.y == 219,
		      "virtual Sense sticks lost input or crossed hands");
		for (bool down: {true, false}) {
			SetButton(HOST_INPUT_CONTROLLER_ID, PAD_BUTTON_CROSS, down);
			Check((ReadSense(left).buttons & 0x4000u) == 0 &&
			          ((ReadSense(right).buttons & 0x4000u) != 0) == down,
			      "native Cross press/release did not remain on the right Sense");
		}
		// Low physical travel sets the digital button too, but its analog
		// event must retain the actual travel across the old 32/33 boundary.
		for (int travel: {1, 32, 33}) {
			const float normalized = travel / 255.0f;
			SetAxis(1, Axis::TriggerLeft, travel);
			check_triggers(normalized, 0.0f);
			SetAxis(1, Axis::TriggerRight, travel);
			check_triggers(normalized, normalized);
			SetAxis(1, Axis::TriggerLeft, 0);
			check_triggers(0.0f, normalized);
			SetAxis(1, Axis::TriggerRight, 0);
			check_triggers(0.0f, 0.0f);
		}
		SetAxis(1, Axis::TriggerLeft, 102);
		SetAxis(1, Axis::TriggerRight, 204);
		check_triggers(0.4f, 0.8f);
		SetAxis(1, Axis::TriggerLeft, 0);
		check_triggers(0.0f, 0.8f);
		SetAxis(1, Axis::TriggerRight, 0);
		check_triggers(0.0f, 0.0f);
		for (uint32_t button: {PAD_BUTTON_L2, PAD_BUTTON_R2}) {
			SetButton(HOST_INPUT_CONTROLLER_ID, button, true);
			check_triggers(button == PAD_BUTTON_L2 ? 1.0f : 0.0f,
			               button == PAD_BUTTON_R2 ? 1.0f : 0.0f);
			SetButton(HOST_INPUT_CONTROLLER_ID, button, false);
			check_triggers(0.0f, 0.0f);
		}
		Check(PadClose(left) == OK && PadClose(right) == OK,
		      "virtual Sense ports did not close");
	}
	Config::Load(Config::ConfigOptions {});
}

void TestSettingCycles() {
	Controller controller;
	for (auto setting:
	     {Setting::SpeakerVolume, Setting::VibrationIntensity, Setting::TriggerEffectIntensity}) {
		Check(GetSettingScale(setting) == 1.0f, "initial setting is not strong");
		const std::vector<float> levels = setting == Setting::SpeakerVolume
		                                      ? std::vector<float> {0.0f, 0.02f, 0.12f, 0.42f, 1.0f}
		                                      : std::vector<float> {0.0f, 0.33f, 0.66f, 1.0f};
		for (int cycle = 0; cycle < 2; ++cycle) {
			for (float level: levels) {
				CycleSetting(setting);
				Check(GetSettingScale(setting) == level, "setting did not cycle or wrap");
			}
		}
	}
	CycleSetting(Setting::SpeakerVolume);
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
}

void TestGlobalControllerLevels() {
	Config::ConfigOptions options;
	options.controller_speaker_volume      = 25;
	options.controller_vibration_intensity = 25;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.5f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.25f &&
		          GetSettingScale(Setting::TriggerEffectIntensity) == 1.0f,
		      "global controller levels did not scale their outputs independently");
		SetRumble(200, 100);
		Check(rumble.back().large == 50 * 257 && rumble.back().small == 25 * 257,
		      "global vibration level did not scale rumble");
		CycleSetting(Setting::SpeakerVolume);
		CycleSetting(Setting::VibrationIntensity);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0f,
		      "cycle hotkeys did not mute globally scaled outputs");
		CycleSetting(Setting::SpeakerVolume);
		CycleSetting(Setting::VibrationIntensity);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.01f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0825f,
		      "cycle hotkeys did not multiply the global controller levels");
		Check(rumble.back().large == 17 * 257 && rumble.back().small == 8 * 257,
		      "cycling vibration did not apply the combined intensity to cached rumble");
	}
	options.controller_speaker_volume = 50;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 1.0f,
		      "speaker midpoint did not preserve the previous default level");
	}
	options.controller_speaker_volume = 100;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 2.0f,
		      "speaker maximum did not add 6 dB of gain");
		CycleSetting(Setting::SpeakerVolume);
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f,
		      "speaker maximum prevented the hotkey from muting");
	}
	options.controller_speaker_volume      = 0;
	options.controller_vibration_intensity = 0;
	Config::Load(options);
	{
		Controller controller;
		Check(GetSettingScale(Setting::SpeakerVolume) == 0.0f &&
		          GetSettingScale(Setting::VibrationIntensity) == 0.0f,
		      "zero global controller levels did not mute outputs");
		SetRumble(200, 100);
		Check(rumble.back().large == 0 && rumble.back().small == 0,
		      "zero global vibration level did not mute motors");
	}
	Config::Load(Config::ConfigOptions {});
}

void TestVibrationLifetime() {
	Controller controller;
	SetRumble(255, 1);
	Check(rumble.size() == 1 && rumble.back().large == 65535 && rumble.back().small == 257 &&
	          rumble.back().duration == 65535,
	      "full vibration changed");
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.back().large == 0 && rumble.back().small == 0, "muting did not stop vibration");
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.back().large == 84 * 257 && rumble.back().small == 257 &&
	          rumble.back().duration == 65335,
	      "changing intensity lost a small motor or extended the vibration deadline");
	Check(effects.empty(), "vibration intensity resent triggers");
	now              = 1000 + 65535;
	const auto calls = rumble.size();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.size() == calls || (rumble.back().large == 0 && rumble.back().small == 0),
	      "expired vibration was resurrected");
	SetRumble(0, 0);
	const auto stopped = rumble.size();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.size() == stopped || (rumble.back().large == 0 && rumble.back().small == 0),
	      "explicitly stopped vibration was resurrected");
}

void TestMaskedTriggersAndValidation() {
	Controller            controller;
	PadTriggerEffectParam param {};
	param.trigger_mask       = 3;
	param.command[0].mode    = 1;
	param.command[0].data[0] = 2;
	param.command[0].data[1] = 8;
	param.command[1].mode    = 2;
	param.command[1].data[0] = 2;
	param.command[1].data[1] = 7;
	param.command[1].data[2] = 6;
	Check(PadSetTriggerEffect(1, &param) == 0, "initial trigger request failed");
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(LastEffect().left_trigger[0] == 5 && LastEffect().right_trigger[0] == 5,
	      "muting did not disable both triggers");
	for (uint32_t mode = 1; mode <= 7; ++mode) {
		auto invalid            = param;
		invalid.command[1]      = {};
		invalid.command[1].mode = mode;
		std::memset(invalid.command[1].data, 255, sizeof(invalid.command[1].data));
		const auto calls = effects.size();
		Check(PadSetTriggerEffect(1, &invalid) == PAD_ERROR_INVALID_ARG && effects.size() == calls,
		      "muting bypassed trigger validation or sent a partial invalid request");
	}
	param.trigger_mask = 4;
	Check(PadSetTriggerEffect(1, &param) == PAD_ERROR_INVALID_ARG, "invalid trigger mask accepted");
	param.trigger_mask       = 1;
	param.command[0].data[1] = 6;
	Check(PadSetTriggerEffect(1, &param) == 0 && LastEffect().enable_bits == 8,
	      "left-only update touched the right trigger");
	CycleSetting(Setting::TriggerEffectIntensity);
	auto effect = LastEffect();
	Check(effect.enable_bits == 12 && effect.left_trigger[0] == 0x21 &&
	          ZoneStrength(effect.left_trigger, 1) == 0 &&
	          ZoneStrength(effect.left_trigger, 2) == 2 && effect.right_trigger[0] == 0x25 &&
	          effect.right_trigger[1] == 0x84 && effect.right_trigger[3] == 1,
	      "masked updates lost a cached trigger or failed to scale its strength");
	CycleSetting(Setting::TriggerEffectIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	effect = LastEffect();
	Check(ZoneStrength(effect.left_trigger, 2) == 6 && effect.right_trigger[3] == 5,
	      "restoring strong intensity did not recover original trigger strengths");
	Check(rumble.empty() && haptics.empty(), "trigger intensity resent vibration");
}

void TestIndependentOutputsAndPadSwitch() {
	Controller controller;
	haptics_handles_rumble = true;
	SetRumble(200, 100);
	now += 100;
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::VibrationIntensity);
	Check(rumble.empty() && haptics.back().large == 66 && haptics.back().small == 33 &&
	          haptics.back().duration == 65435,
	      "haptics rumble did not receive scaled motors and remaining duration");
	PadTriggerEffectParam param {};
	param.trigger_mask       = 1;
	param.command[0].mode    = 1;
	param.command[0].data[1] = 8;
	Check(PadSetTriggerEffect(1, &param) == 0, "trigger request failed");
	const auto vibration_calls = haptics.size();
	const auto trigger_calls   = effects.size();
	CycleSetting(Setting::SpeakerVolume);
	Check(haptics.size() == vibration_calls && effects.size() == trigger_calls,
	      "speaker volume resent controller effects");
	Connect(2);
	Disconnect(1);
	Check(GetActiveControllerId() == 2, "active controller did not change");
	haptics.clear();
	effects.clear();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(effects.empty(), "previous controller's trigger effects leaked to new pad");
	for (const auto& request: haptics) {
		Check(request.large == 0 && request.small == 0,
		      "previous controller's vibration leaked to new pad");
	}
	SetRumble(200, 100);
	Check(PadSetTriggerEffect(1, &param) == 0, "replacement pad trigger request failed");
	EmergencyShutdown();
	Check(GetActiveControllerId() == -1, "released controller remained active");
	haptics.clear();
	effects.clear();
	CycleSetting(Setting::VibrationIntensity);
	CycleSetting(Setting::TriggerEffectIntensity);
	Check(effects.empty(), "released controller received a trigger request");
	for (const auto& request: haptics) {
		Check(request.large == 0 && request.small == 0,
		      "released controller retained a cached vibration");
	}
}

void TestTriggerEffectState() {
	Controller                       controller;
	PadTriggerEffectStateInformation info {{-1, -1}};
	const auto check_state = [&](int32_t left, int32_t right, const char* message) {
		Check(PadGetTriggerEffectState(1, &info) == OK && info.state[0] == left &&
		          info.state[1] == right,
		      message);
	};
	const auto axis = [&](int left, int right) {
		SetAxis(1, Axis::TriggerLeft, left);
		SetAxis(1, Axis::TriggerRight, right);
	};
	PadTriggerEffectParam param {};
	const auto            set_effect = [&] {
		Check(PadSetTriggerEffect(1, &param) == OK, "trigger request failed");
	};
	Check(PadGetTriggerEffectState(2, &info) == PAD_ERROR_INVALID_HANDLE && info.state[0] == -1 &&
	          info.state[1] == -1,
	      "invalid trigger handle changed the output");
	Check(PadGetTriggerEffectState(2, nullptr) == PAD_ERROR_INVALID_HANDLE,
	      "trigger state checked the output before the handle");
	Check(PadGetTriggerEffectState(1, nullptr) == PAD_ERROR_INVALID_ARG,
	      "trigger state accepted a null output");
	check_state(0, 0, "unset trigger effects did not report off");

	param.trigger_mask       = 3;
	param.command[0].mode    = 1;
	param.command[0].data[0] = 7;
	param.command[0].data[1] = 8;
	param.command[1].mode    = 2;
	param.command[1].data[0] = 2;
	param.command[1].data[1] = 7;
	param.command[1].data[2] = 6;
	set_effect();
	axis(191, 63);
	check_state(1, 3, "triggers activated before the documented start positions");
	axis(192, 64);
	check_state(2, 4, "triggers did not activate at the documented start positions");
	axis(255, 223);
	check_state(2, 4, "weapon fired before its end position");
	SetAxis(1, Axis::TriggerRight, 224);
	SetAxis(1, Axis::TriggerRight, 100);
	check_state(2, 5, "unpolled weapon firing was lost on partial release");
	check_state(2, 5, "reading trigger state changed the weapon latch");
	set_effect();
	check_state(2, 5, "repeating a weapon request rearmed a held trigger");
	CycleSetting(Setting::TriggerEffectIntensity);
	check_state(2, 5, "muting trigger effects changed their reported states");

	// A rejected two-trigger request must leave both cached commands and states intact.
	auto invalid               = param;
	invalid.command[0].mode    = 0;
	invalid.command[1].data[1] = 2;
	Check(PadSetTriggerEffect(1, &invalid) == PAD_ERROR_INVALID_ARG,
	      "invalid weapon end position was accepted");
	SetAxis(1, Axis::TriggerLeft, 255);
	check_state(2, 5, "an invalid request changed cached trigger commands or states");
	SetAxis(1, Axis::TriggerRight, 63);
	SetAxis(1, Axis::TriggerRight, 100);
	check_state(2, 4, "unpolled release did not rearm the weapon");
	param.trigger_mask       = 2;
	param.command[1].data[1] = 8;
	set_effect();
	axis(255, 253);
	check_state(2, 4, "weapon end position 8 fired before output 254");
	SetAxis(1, Axis::TriggerRight, 254);
	check_state(2, 5, "weapon end position 8 did not fire at output 254");

	// Masked updates preserve the other trigger, including its fired state.
	param.trigger_mask       = 1;
	param.command[0].data[0] = 9;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 254);
	check_state(1, 5, "feedback position 9 activated before full travel");
	SetAxis(HOST_INPUT_CONTROLLER_ID, Axis::TriggerLeft, 255);
	check_state(2, 5, "keyboard trigger travel did not activate feedback");
	param.command[0].data[0] = 0;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 0);
	check_state(2, 5, "feedback at position 0 was suppressed by a dead zone");
	param.command[0].data[1] = 0;
	set_effect();
	check_state(1, 5, "zero-strength feedback reported active");

	param.command[0]         = {};
	param.command[0].mode    = 4;
	param.command[0].data[0] = 4;
	param.command[0].data[2] = 4;
	param.command[0].data[8] = 4;
	set_effect();
	const int feedback_cases[][2] = {{0, 2},  {1, 1},   {31, 1},  {32, 2},  {63, 2},
	                                 {64, 1}, {223, 1}, {224, 2}, {254, 2}, {255, 1}};
	for (const auto& test: feedback_cases) {
		SetAxis(1, Axis::TriggerLeft, test[0]);
		check_state(test[1], 5, "multi-position feedback used the wrong position");
	}

	param.command[0]         = {};
	param.command[0].mode    = 5;
	param.command[0].data[0] = 3;
	param.command[0].data[1] = 7;
	param.command[0].data[2] = 2;
	param.command[0].data[3] = 6;
	set_effect();
	const int slope_cases[][2] = {{63, 1}, {64, 2}, {191, 2}, {192, 1}, {255, 1}};
	for (const auto& test: slope_cases) {
		SetAxis(1, Axis::TriggerLeft, test[0]);
		check_state(test[1], 5, "slope feedback did not respect its start and end");
	}
	param.command[0].data[0] = 0;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 0);
	check_state(2, 5, "slope feedback at position 0 was suppressed");

	param.command[0]         = {};
	param.command[0].mode    = 3;
	param.command[0].data[0] = 6;
	param.command[0].data[1] = 4;
	param.command[0].data[2] = 20;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 159);
	check_state(6, 5, "vibration activated before position 6");
	SetAxis(1, Axis::TriggerLeft, 160);
	check_state(7, 5, "vibration did not activate at position 6");
	param.command[0].data[0] = 0;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 0);
	check_state(7, 5, "vibration at position 0 was suppressed");
	param.command[0].data[2] = 0;
	set_effect();
	check_state(6, 5, "zero-frequency vibration reported active");

	param.command[0]          = {};
	param.command[0].mode     = 6;
	param.command[0].data[0]  = 20;
	param.command[0].data[10] = 4;
	set_effect();
	SetAxis(1, Axis::TriggerLeft, 254);
	check_state(6, 5, "multi-position vibration activated before position 9");
	SetAxis(1, Axis::TriggerLeft, 255);
	check_state(7, 5, "multi-position vibration did not activate at position 9");
	param.command[0].data[0] = 0;
	set_effect();
	check_state(6, 5, "zero-frequency multi-position vibration reported active");

	ResetInputState();
	check_state(6, 3, "input reset retained a fired weapon");
	param.command[0].mode = 0;
	set_effect();
	check_state(0, 3, "off mode retained an effect state");
	Connect(2);
	Disconnect(1);
	check_state(0, 0, "switching controllers retained effect states");
}

} // namespace

int main(int argc, char** argv) {
	const ScopedEnvironment no_hand_grid("KYTY_VR_AIM_GRID", nullptr);
	const ScopedEnvironment no_gaze_grid("KYTY_VR_GAZE_GRID", nullptr);
	const ScopedEnvironment world_hands("KYTY_VR_HAND_LOCAL_ORIGIN", nullptr);
	const ScopedEnvironment left_yaw("KYTY_VR_LEFT_AIM_YAW", nullptr);
	const ScopedEnvironment right_yaw("KYTY_VR_RIGHT_AIM_YAW", nullptr);
	const ScopedEnvironment left_pitch("KYTY_VR_LEFT_AIM_PITCH", nullptr);
	const ScopedEnvironment right_pitch("KYTY_VR_RIGHT_AIM_PITCH", nullptr);
	Config::Initialize();
	// InputMap is intentionally immutable for the process. Run the custom-map
	// variant in a fresh process, without changing production keymap lifetime.
	if (argc == 2 && std::strcmp(argv[1], "--custom-vr-hand-bindings") == 0) {
		TestDesktopHandBindings(true);
		Config::Shutdown();
		return 0;
	}
	TestPadPortOwnership();
	TestVirtualSenseInput();
	TestIndependentHandPositions();
	TestDesktopHandBindings(false);
	TestSettingCycles();
	TestGlobalControllerLevels();
	// Each following test initializes another controller and requires strong defaults.
	TestVibrationLifetime();
	TestMaskedTriggersAndValidation();
	TestIndependentOutputsAndPadSwitch();
	TestTriggerEffectState();
	Config::Shutdown();
	return 0;
}
