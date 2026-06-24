# file: agent/error_handling.py

import functools

MAX_RETRIES = 3

def with_error_handling(node_name: str):

    def decorator(node_func):
        @functools.wraps(node_func)
        def wrapped_node(state: dict) -> dict:
            try:
                # Run the original node function completely normally.
                # If it succeeds, its return dict flows through untouched.
                result = node_func(state)
                return result
            except Exception as e:
                # Capture the error rather than letting it propagate.
                # We do NOT re-raise here, we want LangGraph to continue executing so our routing function gets a chance to decide whether to retry or give up gracefully.
                print(f"[{node_name}] ERROR caught: {e}")
                return {
                    "current_step":  node_name,
                    "error_message": str(e),
                    # Increment retry_count so the routing function knows how many times this step has already failed.
                    "retry_count":   state.get("retry_count", 0) + 1
                }
        return wrapped_node
    return decorator


def route_after_error_check(state: dict) -> str:
    
    if state.get("error_message") is None:
        return "continue"

    if state.get("retry_count", 0) < MAX_RETRIES:
        print(f"  Retrying {state['current_step']} "
              f"(attempt {state['retry_count'] + 1}/{MAX_RETRIES})")
        return "retry"

    print(f"  Giving up on {state['current_step']} "
          f"after {MAX_RETRIES} retries")
    return "give_up"