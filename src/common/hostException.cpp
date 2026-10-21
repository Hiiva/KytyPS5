#include "common/hostException.h"

#include <atomic>
#include <cstdio>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#include <dbghelp.h>
#include <cwchar>
#include <initializer_list>
#else
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <initializer_list>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/ucontext.h>
#else
#include <ucontext.h> // IWYU pragma: keep
#endif
#endif

// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <excpt.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <wtypes.h>

namespace Common::HostException {

#if !defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX

// macOS uses the same POSIX platform setting and needs a signal stack too.
class ThreadSignalStack {
public:
	ThreadSignalStack() {
		const auto page_size = static_cast<size_t>(::getpagesize());
		const auto stack_size =
		    (std::max<size_t>(64 * 1024, MINSIGSTKSZ) + page_size - 1) & ~(page_size - 1);
		if (::posix_memalign(&m_memory, page_size, stack_size) != 0) {
			return;
		}

		stack_t stack {};
		stack.ss_sp   = m_memory;
		stack.ss_size = stack_size;
		if (::sigaltstack(&stack, &m_previous) != 0) {
			std::free(m_memory);
			m_memory = nullptr;
		}
	}

	~ThreadSignalStack() {
		if (m_memory != nullptr && ::sigaltstack(&m_previous, nullptr) == 0) {
			std::free(m_memory);
		}
	}

	[[nodiscard]] bool IsInitialized() const { return m_memory != nullptr; }

	KYTY_CLASS_NO_COPY(ThreadSignalStack)

private:
	void*   m_memory = nullptr;
	stack_t m_previous {};
};

bool InitializeThreadSignalStack() {
	// Keep fault handling off guest stacks, which GPU tracking can make read-only.
	thread_local ThreadSignalStack signal_stack;
	return signal_stack.IsInitialized();
}

#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

struct NativeFault {
	EXCEPTION_RECORD record {};
	CONTEXT context {};
};

struct CrashCapture {
	std::atomic<bool> enabled {false};
	std::atomic<bool> claimed {false};
	HANDLE request = nullptr;
	HANDLE complete = nullptr;
	HANDLE report = INVALID_HANDLE_VALUE;
	HANDLE dump = INVALID_HANDLE_VALUE;
	HMODULE dbghelp = nullptr;
	HMODULE dbgcore = nullptr;
	decltype(&MiniDumpWriteDump) write_dump = nullptr;
	LPTOP_LEVEL_EXCEPTION_FILTER previous_filter = nullptr;
	DWORD worker_id = 0;
	DWORD fault_thread_id = 0;
	NativeFault fault {};
};

CrashCapture g_crash;
thread_local bool g_in_exception_handler = false;
thread_local NativeFault g_outer_fault {};

NativeFault CopyFault(const EXCEPTION_POINTERS& exception) noexcept {
	NativeFault fault {*exception.ExceptionRecord, *exception.ContextRecord};
	// The chained record can belong to the interrupted stack.
	fault.record.ExceptionRecord = nullptr;
	return fault;
}

struct CrashText {
	char data[2048] {};
	DWORD length = 0;

	void Text(const char* text) noexcept {
		while (*text != '\0' && length < sizeof(data)) {
			data[length++] = *text++;
		}
	}
	void Hex(const char* name, uint64_t value) noexcept {
		Text(name);
		Text("=0x");
		for (int shift = 60; shift >= 0; shift -= 4) {
			const char digit[2] = {"0123456789abcdef"[(value >> shift) & 15u], '\0'};
			Text(digit);
		}
		Text("\r\n");
	}
	void Write() const noexcept {
		DWORD written = 0;
		WriteFile(g_crash.report, data, length, &written, nullptr);
		FlushFileBuffers(g_crash.report);
	}
};

void ReportFault(CrashText& text, const NativeFault& fault) noexcept {
	text.Hex("exception", fault.record.ExceptionCode);
	text.Hex("pc", fault.context.Rip);
	text.Hex("exception_address", reinterpret_cast<uint64_t>(fault.record.ExceptionAddress));
	if (fault.record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
	    fault.record.NumberParameters >= 2) {
		text.Hex("access", fault.record.ExceptionInformation[0]);
		text.Hex("fault_address", fault.record.ExceptionInformation[1]);
	}
	text.Hex("rax", fault.context.Rax);
	text.Hex("rbx", fault.context.Rbx);
	text.Hex("rcx", fault.context.Rcx);
	text.Hex("rdx", fault.context.Rdx);
	text.Hex("rsi", fault.context.Rsi);
	text.Hex("rdi", fault.context.Rdi);
	text.Hex("rbp", fault.context.Rbp);
	text.Hex("rsp", fault.context.Rsp);
	text.Hex("r8", fault.context.R8);
	text.Hex("r9", fault.context.R9);
	text.Hex("r10", fault.context.R10);
	text.Hex("r11", fault.context.R11);
	text.Hex("r12", fault.context.R12);
	text.Hex("r13", fault.context.R13);
	text.Hex("r14", fault.context.R14);
	text.Hex("r15", fault.context.R15);
}

DWORD WINAPI CrashDumpWorker(void*) noexcept {
	if (WaitForSingleObject(g_crash.request, INFINITE) != WAIT_OBJECT_0) {
		return 0;
	}
	CrashText result;
	MEMORY_BASIC_INFORMATION memory {};
	if (VirtualQuery(reinterpret_cast<const void*>(g_crash.fault.context.Rip),
	                 &memory, sizeof(memory)) != 0) {
		result.Hex("pc_allocation_base", reinterpret_cast<uint64_t>(memory.AllocationBase));
		result.Hex("pc_memory_type", memory.Type);
	}
	result.Write();
	EXCEPTION_POINTERS pointers {&g_crash.fault.record, &g_crash.fault.context};
	MINIDUMP_EXCEPTION_INFORMATION exception {g_crash.fault_thread_id, &pointers, FALSE};
	const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpNormal | MiniDumpWithThreadInfo);
	const BOOL ok = g_crash.write_dump(GetCurrentProcess(), GetCurrentProcessId(), g_crash.dump,
	                                  type, &exception, nullptr, nullptr);
	const DWORD error = ok ? ERROR_SUCCESS : GetLastError();
	FlushFileBuffers(g_crash.dump);
	CrashText status;
	status.Hex("dump_success", ok != FALSE);
	status.Hex("dump_error", error);
	status.Write();
	SetEvent(g_crash.complete);
	return 0;
}

void CaptureNativeFault(EXCEPTION_POINTERS* exception) noexcept {
	if (!g_crash.enabled.load(std::memory_order_acquire) || exception == nullptr ||
	    exception->ExceptionRecord == nullptr || exception->ContextRecord == nullptr) {
		return;
	}
	bool expected = false;
	if (g_crash.claimed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
		g_crash.fault_thread_id = GetCurrentThreadId();
		g_crash.fault = CopyFault(*exception);
		// Save the indispensable evidence without CRT logging, guest TLS or DbgHelp.
		CrashText report;
		report.Text("Kyty Windows crash capture v1\r\n");
		report.Hex("pid", GetCurrentProcessId());
		report.Hex("thread_id", g_crash.fault_thread_id);
		ReportFault(report, g_crash.fault);
		report.Write();
		if (g_in_exception_handler &&
		    (g_outer_fault.context.Rip != g_crash.fault.context.Rip ||
		     g_outer_fault.context.Rsp != g_crash.fault.context.Rsp)) {
			CrashText outer;
			outer.Text("Original exception before nested recovery fault:\r\n");
			ReportFault(outer, g_outer_fault);
			outer.Write();
		}
		SetEvent(g_crash.request);
	}
	if (GetCurrentThreadId() != g_crash.worker_id &&
	    WaitForSingleObject(g_crash.complete, 3000) == WAIT_TIMEOUT) {
		CrashText timeout;
		timeout.Text("dump_wait_timeout=1\r\n");
		timeout.Write();
	}
}

LONG WINAPI CrashUnhandledFilter(EXCEPTION_POINTERS* exception) noexcept {
	CaptureNativeFault(exception);
	const auto previous = g_crash.previous_filter;
	return previous != nullptr ? previous(exception) : EXCEPTION_CONTINUE_SEARCH;
}

void InitializeCrashCapture() {
	wchar_t prefix[2048] {};
	const DWORD size = GetEnvironmentVariableW(L"KYTY_CRASH_DUMP_PREFIX", prefix, 2048);
	if (size == 0) {
		return;
	}
	if (size >= 2048) {
		std::printf("Windows crash capture: dump prefix is too long\n");
		return;
	}
	wchar_t report_path[2176] {}, dump_path[2176] {};
	const auto pid = GetCurrentProcessId();
	const auto tick = GetTickCount64();
	std::swprintf(report_path, 2176, L"%ls-%lu-%016llx.crash.txt", prefix, pid, tick);
	std::swprintf(dump_path, 2176, L"%ls-%lu-%016llx.dmp", prefix, pid, tick);
	g_crash.dbghelp = LoadLibraryExW(L"dbghelp.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
	g_crash.dbgcore = LoadLibraryExW(L"dbgcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
	if (g_crash.dbghelp != nullptr) {
		g_crash.write_dump = reinterpret_cast<decltype(g_crash.write_dump)>(
		    GetProcAddress(g_crash.dbghelp, "MiniDumpWriteDump"));
	}
	if (g_crash.write_dump != nullptr && g_crash.dbgcore != nullptr) {
		g_crash.report = CreateFileW(report_path, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
		                            CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
		g_crash.dump = CreateFileW(dump_path, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
		                          CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
		g_crash.request = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		g_crash.complete = CreateEventW(nullptr, TRUE, FALSE, nullptr);
		if (g_crash.report != INVALID_HANDLE_VALUE && g_crash.dump != INVALID_HANDLE_VALUE &&
		    g_crash.request != nullptr && g_crash.complete != nullptr) {
			if (HANDLE worker = CreateThread(nullptr, 0, CrashDumpWorker, nullptr, 0,
			                                 &g_crash.worker_id); worker != nullptr) {
				CloseHandle(worker);
				g_crash.previous_filter = SetUnhandledExceptionFilter(CrashUnhandledFilter);
				g_crash.enabled.store(true, std::memory_order_release);
				std::printf("Windows crash capture: %ls\n", dump_path);
				std::fflush(stdout);
				return;
			}
		}
	}
	const DWORD error = GetLastError();
	for (HANDLE handle: {g_crash.report, g_crash.dump, g_crash.request, g_crash.complete}) {
		if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
			CloseHandle(handle);
		}
	}
	if (g_crash.dbgcore != nullptr) {
		FreeLibrary(g_crash.dbgcore);
	}
	if (g_crash.dbghelp != nullptr) {
		FreeLibrary(g_crash.dbghelp);
	}
	std::printf("Windows crash capture initialization failed: %lu\n", error);
	std::fflush(stdout);
}

} // namespace

static LONG WINAPI ExceptionFilter(PEXCEPTION_POINTERS exception) noexcept {
	auto* exception_record = exception->ExceptionRecord;

	if (exception_record->ExceptionCode == DBG_PRINTEXCEPTION_C ||
	    exception_record->ExceptionCode == DBG_PRINTEXCEPTION_WIDE_C) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	if (exception_record->ExceptionCode == 0x406D1388) {
		// Set a thread name.
		return EXCEPTION_CONTINUE_EXECUTION;
	}

	ExceptionInfo info {};
	info.exception_address = reinterpret_cast<uint64_t>(exception_record->ExceptionAddress);
	info.native_code       = exception_record->ExceptionCode;
	info.native_context    = exception->ContextRecord;
	info.native_exception_record = exception_record;

	if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
		info.type = ExceptionType::AccessViolation;
		switch (exception_record->ExceptionInformation[0]) {
			case 0: info.access_violation_type = AccessViolationType::Read; break;
			case 1: info.access_violation_type = AccessViolationType::Write; break;
			case 8: info.access_violation_type = AccessViolationType::Execute; break;
			default: info.access_violation_type = AccessViolationType::Unknown; break;
		}
		info.access_violation_vaddr = exception_record->ExceptionInformation[1];
	} else if (exception_record->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	info.rax = exception->ContextRecord->Rax;
	info.rbx = exception->ContextRecord->Rbx;
	info.rcx = exception->ContextRecord->Rcx;
	info.rdx = exception->ContextRecord->Rdx;
	info.rsi = exception->ContextRecord->Rsi;
	info.rdi = exception->ContextRecord->Rdi;
	info.rbp = exception->ContextRecord->Rbp;
	info.rsp = exception->ContextRecord->Rsp;
	info.r8  = exception->ContextRecord->R8;
	info.r9  = exception->ContextRecord->R9;
	info.r10 = exception->ContextRecord->R10;
	info.r11 = exception->ContextRecord->R11;
	info.r12 = exception->ContextRecord->R12;
	info.r13 = exception->ContextRecord->R13;
	info.r14 = exception->ContextRecord->R14;
	info.r15 = exception->ContextRecord->R15;

	const bool capture_enabled = g_crash.enabled.load(std::memory_order_acquire);
	if (capture_enabled &&
	    (g_in_exception_handler || GetCurrentThreadId() == g_crash.worker_id)) {
		// Leave nested recovery faults and DbgHelp's own guarded probes to Windows
		// instead of recursively invoking guest recovery or thread diagnostics.
		return EXCEPTION_CONTINUE_SEARCH;
	}
	if (capture_enabled) {
		g_outer_fault = CopyFault(*exception);
		g_in_exception_handler = true;
	}
	const auto handler = g_handler.load(std::memory_order_acquire);
	const bool handled = handler != nullptr && handler(info);
	if (capture_enabled) {
		g_in_exception_handler = false;
	}
	if (handled) {
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	return EXCEPTION_CONTINUE_SEARCH;
}

#elif defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);

// Translate the x86-64 page-fault error code (mcontext __es.__err) into an access type.
// bit 1 (0x2) = write, bit 4 (0x10) = instruction fetch, otherwise a read.
static AccessViolationType DecodeAccess(uint64_t err) {
	if ((err & 0x10u) != 0) {
		return AccessViolationType::Execute;
	}
	if ((err & 0x2u) != 0) {
		return AccessViolationType::Write;
	}
	return AccessViolationType::Read;
}

// POSIX signal handler that mirrors the Windows vectored handler: build an ExceptionInfo
// from the mcontext and dispatch. A resolved fault (handler returns true) simply returns,
// re-executing the faulting instruction against the now-fixed protection. An unresolved
// fault restores the default disposition so the retry terminates the process.
static void SignalHandler(int sig, siginfo_t* si, void* uctx) {
	auto*       uc = static_cast<ucontext_t*>(uctx);
	const auto* mc = uc->uc_mcontext;
	const auto& ss = mc->__ss;

	ExceptionInfo info {};
	info.exception_address = ss.__rip;
	info.native_code       = static_cast<uint32_t>(si->si_code);
	info.native_context    = uctx;

	if (sig == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		info.type                   = ExceptionType::AccessViolation;
		info.access_violation_type  = DecodeAccess(mc->__es.__err);
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(si->si_addr);
	}

	info.rax = ss.__rax;
	info.rbx = ss.__rbx;
	info.rcx = ss.__rcx;
	info.rdx = ss.__rdx;
	info.rsi = ss.__rsi;
	info.rdi = ss.__rdi;
	info.rbp = ss.__rbp;
	info.rsp = ss.__rsp;
	info.r8  = ss.__r8;
	info.r9  = ss.__r9;
	info.r10 = ss.__r10;
	info.r11 = ss.__r11;
	info.r12 = ss.__r12;
	info.r13 = ss.__r13;
	info.r14 = ss.__r14;
	info.r15 = ss.__r15;

	const auto handler = g_handler.load(std::memory_order_acquire);
	if (handler != nullptr && handler(info)) {
		return; // retry the faulting instruction against the fixed mapping
	}

	// Unresolved: restore the default action so the re-executed instruction terminates.
	struct sigaction dfl {};
	dfl.sa_handler = SIG_DFL;
	sigemptyset(&dfl.sa_mask);
	sigaction(sig, &dfl, nullptr);
}

#else

// x86-64 page-fault error bits.
constexpr uint64_t PAGE_FAULT_ERROR_WRITE       = 0x02;
constexpr uint64_t PAGE_FAULT_ERROR_INSTRUCTION = 0x10;

// Let the kernel handle an unresolved fault on retry.
static void ChainToDefault(int signal_number) noexcept {
	struct sigaction restore {};
	restore.sa_handler = SIG_DFL;
	sigemptyset(&restore.sa_mask);
	restore.sa_flags = 0;
	::sigaction(signal_number, &restore, nullptr);
}

static void SignalHandler(int signal_number, siginfo_t* signal_info, void* native_context) {
	auto* context = static_cast<ucontext_t*>(native_context);
	auto* gregs   = context->uc_mcontext.gregs;

	ExceptionInfo info {};
	info.exception_address = static_cast<uint64_t>(gregs[REG_RIP]);
	info.native_code       = static_cast<uint32_t>(signal_number);
	info.native_context    = context;

	if (signal_number == SIGSEGV || signal_number == SIGBUS) {
		info.type             = ExceptionType::AccessViolation;
		const auto error_code = static_cast<uint64_t>(gregs[REG_ERR]);
		if ((error_code & PAGE_FAULT_ERROR_INSTRUCTION) != 0) {
			info.access_violation_type = AccessViolationType::Execute;
		} else if ((error_code & PAGE_FAULT_ERROR_WRITE) != 0) {
			info.access_violation_type = AccessViolationType::Write;
		} else {
			info.access_violation_type = AccessViolationType::Read;
		}
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(signal_info->si_addr);
	} else if (signal_number == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		ChainToDefault(signal_number);
		return;
	}

	info.rax = static_cast<uint64_t>(gregs[REG_RAX]);
	info.rbx = static_cast<uint64_t>(gregs[REG_RBX]);
	info.rcx = static_cast<uint64_t>(gregs[REG_RCX]);
	info.rdx = static_cast<uint64_t>(gregs[REG_RDX]);
	info.rsi = static_cast<uint64_t>(gregs[REG_RSI]);
	info.rdi = static_cast<uint64_t>(gregs[REG_RDI]);
	info.rbp = static_cast<uint64_t>(gregs[REG_RBP]);
	info.rsp = static_cast<uint64_t>(gregs[REG_RSP]);
	info.r8  = static_cast<uint64_t>(gregs[REG_R8]);
	info.r9  = static_cast<uint64_t>(gregs[REG_R9]);
	info.r10 = static_cast<uint64_t>(gregs[REG_R10]);
	info.r11 = static_cast<uint64_t>(gregs[REG_R11]);
	info.r12 = static_cast<uint64_t>(gregs[REG_R12]);
	info.r13 = static_cast<uint64_t>(gregs[REG_R13]);
	info.r14 = static_cast<uint64_t>(gregs[REG_R14]);
	info.r15 = static_cast<uint64_t>(gregs[REG_R15]);

	const auto handler = g_handler.load(std::memory_order_acquire);
	if (handler != nullptr && handler(info)) {
		return;
	}

	ChainToDefault(signal_number);
}

#endif

void CaptureFatalException(const ExceptionInfo& info) noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	EXCEPTION_POINTERS pointers {
	    static_cast<EXCEPTION_RECORD*>(info.native_exception_record),
	    static_cast<CONTEXT*>(info.native_context)};
	CaptureNativeFault(&pointers);
#else
	(void)info;
#endif
}

bool InstallHandler(Handler handler) {
	if (handler == nullptr) {
		return false;
	}

	uint32_t expected_state = 0;
	if (!g_install_state.compare_exchange_strong(expected_state, 1, std::memory_order_acq_rel)) {
		return expected_state == 2 && g_handler.load(std::memory_order_acquire) == handler;
	}

	g_handler.store(handler, std::memory_order_release);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (AddVectoredExceptionHandler(0, ExceptionFilter) == nullptr) {
		g_handler.store(nullptr, std::memory_order_release);
		g_install_state.store(0, std::memory_order_release);
		printf("AddVectoredExceptionHandler() failed\n");
		return false;
	}
	InitializeCrashCapture();
#elif defined(__APPLE__)
	struct sigaction sa {};
	sa.sa_sigaction = SignalHandler;
	sa.sa_flags     = SA_SIGINFO | SA_ONSTACK;
	sigemptyset(&sa.sa_mask);
	// The guest signal-dispatch path (KernelRaiseException) interrupts threads with
	// SIGUSR1; block it while a fault is being resolved so a stop-the-world request
	// cannot preempt the handler between the protection fix and the retry.
	sigaddset(&sa.sa_mask, SIGUSR1);

	// macOS raises SIGBUS for protection faults on some paths and SIGSEGV on others;
	// SIGILL covers instructions the host cannot execute (routed to the x64 emulator).
	bool ok = sigaction(SIGSEGV, &sa, nullptr) == 0 && sigaction(SIGBUS, &sa, nullptr) == 0 &&
	          sigaction(SIGILL, &sa, nullptr) == 0;
	if (!ok) {
		g_handler.store(nullptr, std::memory_order_release);
		g_install_state.store(0, std::memory_order_release);
		printf("sigaction() failed to install the host fault handler\n");
		return false;
	}
#else
	struct sigaction action {};
	action.sa_sigaction = SignalHandler;
	sigemptyset(&action.sa_mask);
	action.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;

	for (const int signal_number: {SIGSEGV, SIGBUS, SIGILL}) {
		if (::sigaction(signal_number, &action, nullptr) != 0) {
			g_handler.store(nullptr, std::memory_order_release);
			g_install_state.store(0, std::memory_order_release);
			printf("sigaction(%d) failed\n", signal_number);
			return false;
		}
	}
#endif

	g_install_state.store(2, std::memory_order_release);
	return true;
}

} // namespace Common::HostException
