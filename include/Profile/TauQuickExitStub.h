#ifndef _TAU_QUICK_EXIT_STUB_H_
#define _TAU_QUICK_EXIT_STUB_H_

/*
 * Workaround for libstdc++ <cstdlib> that unconditionally references
 * `::at_quick_exit` / `::quick_exit` based on libstdc++'s
 * `_GLIBCXX_HAVE_AT_QUICK_EXIT` / `_GLIBCXX_HAVE_QUICK_EXIT` macros, when the
 * platform's <stdlib.h> does not actually declare those symbols.
 *
 * Observed on macOS with Homebrew GCC 15: every macOS SDK shipped to date
 * (11.x through 26.x) lacks the declarations, but Homebrew GCC's installed
 * `c++config.h` hard-codes both macros to 1. The mismatch causes any C++
 * translation unit that includes <stdlib.h>/<cstdlib> to fail to compile
 * with "'at_quick_exit' has not been declared in '::'".
 *
 * This header provides only declarations -- no definitions. As long as
 * nothing actually calls `quick_exit` / `at_quick_exit`, link succeeds.
 * If something does call them the link fails loudly with an undefined
 * symbol; that is the correct signal because the platform genuinely lacks
 * the implementation.
 *
 * Enabled by configure when the probe in `configure` (which test-compiles
 * `#include <stdlib.h>` with the selected C++ compiler) fails. The
 * configure machinery defines `TAU_NEEDS_QUICK_EXIT_STUB` via the
 * generated stub Makefile so both TAU's own build and user TUs compiled
 * via the `tau_*.sh` wrappers pick up the workaround.
 *
 * Must be included BEFORE any <stdlib.h>/<cstdlib>.
 */

#ifdef TAU_NEEDS_QUICK_EXIT_STUB

#ifdef __cplusplus
extern "C" {
#endif

int at_quick_exit(void (*)(void));
void quick_exit(int) __attribute__((noreturn));

#ifdef __cplusplus
}
#endif

#endif /* TAU_NEEDS_QUICK_EXIT_STUB */

#endif /* _TAU_QUICK_EXIT_STUB_H_ */
