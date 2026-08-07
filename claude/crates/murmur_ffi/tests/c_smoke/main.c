/* A minimal real-C consumer proving murmur_ffi's cbindgen-generated header
 * (include/murmur_ffi.h) actually links and works against the compiled cdylib -- not just the
 * Rust-calling-Rust round-trip murmur_ffi's own #[cfg(test)] suite already covers (Phase 11,
 * roadmap.md). Compiled and run by ../c_smoke.rs via `cc`, not by Cargo directly (Cargo
 * doesn't compile C sources itself). */

#include "murmur_ffi.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    struct MurmurConfig config;
    memset(&config, 0, sizeof(config));
    config.mode = "pearce";
    config.modifier = "instant_response";
    config.domain = "open";
    config.spatial_index = "hash_grid";
    config.neighbor_selection = "radius_gather";
    config.speed_model = "band";
    config.init = "sphere_volume";
    config.noise = "uniform_sphere";
    config.cruise_speed = 1.0;
    config.max_force = 5.0;
    config.speed_min_factor = 0.3;
    config.boid_count = 40;
    config.dt = 1.0;
    config.vision_radius = 10.0;

    struct CKeyValue params[2] = {
        {"cell_size", 10.0},
        {"radius", 25.0},
    };
    config.plugin_params = params;
    config.plugin_params_len = 2;
    config.init_seed = 7;
    config.spawn_headroom = 0;

    /* Composes two StepHooks whose own checkpoint fields this smoke test verifies below --
     * proving the new per-boid/scene-level CCheckpoint fields (design/05_viz_contract.md §2.1/
     * §2.2) are genuinely reachable from real C, not just from murmur_ffi's own Rust tests. */
    const char *hooks[2] = {"boid_state_machine", "ecology"};
    config.step_hooks = hooks;
    config.step_hooks_len = 2;

    struct MurmurSimulation *sim = murmur_create(&config);
    if (sim == NULL) {
        fprintf(stderr, "murmur_create failed: %s\n", murmur_last_error_message());
        return 1;
    }

    if (murmur_boid_count(sim) != 40) {
        fprintf(stderr, "unexpected boid_count\n");
        return 1;
    }

    struct CCommand stride_cmd;
    memset(&stride_cmd, 0, sizeof(stride_cmd));
    stride_cmd.kind = CMD_SET_CHECKPOINT_STRIDE;
    stride_cmd.stride = 5;

    struct MurmurCheckpointBuffer *buffer = NULL;
    int32_t status = murmur_run_batch(sim, 15, 1, &stride_cmd, 1, &buffer);
    if (status != 0) {
        fprintf(stderr, "murmur_run_batch failed with status %d\n", status);
        return 1;
    }

    uintptr_t n = murmur_checkpoint_buffer_len(buffer);
    if (n != 3) {
        fprintf(stderr, "expected 3 checkpoints (15/5), got %zu\n", (size_t)n);
        return 1;
    }

    struct CCheckpoint cp = murmur_checkpoint_buffer_get(buffer, n - 1);
    if (cp.boid_count != 40 || cp.boids == NULL) {
        fprintf(stderr, "unexpected checkpoint boid data\n");
        return 1;
    }
    for (uint32_t i = 0; i < cp.boid_count; i++) {
        double x = cp.boids[i].position.x;
        if (x != x) { /* NaN check -- NaN is the only value that isn't equal to itself */
            fprintf(stderr, "boid %u position.x is NaN\n", i);
            return 1;
        }
        if (!cp.boids[i].has_state) {
            fprintf(stderr, "boid %u has no state -- boid_state_machine's own checkpoint field "
                            "didn't reach C\n", i);
            return 1;
        }
    }
    if (!cp.has_environment) {
        fprintf(stderr, "checkpoint has no environment -- ecology's own checkpoint field "
                        "didn't reach C\n");
        return 1;
    }

    murmur_checkpoint_buffer_destroy(buffer);
    murmur_destroy(sim);

    printf("C_SMOKE_OK\n");
    return 0;
}
