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

    /* cell_size deliberately disagrees with vision_radius (10.0) -- proves the
     * design/01_core.md §4.1 construction-time Warning (HashGrid's own cell_size snapped to
     * vision_radius) reaches real C via murmur_warning_count/murmur_warning_message, not just
     * murmur_core's own Rust tests. */
    struct CKeyValue params[2] = {
        {"cell_size", 3.0},
        {"radius", 25.0},
    };
    config.plugin_params = params;
    config.plugin_params_len = 2;
    config.init_seed = 7;
    config.spawn_headroom = 0;

    /* Composes three StepHooks whose own checkpoint fields this smoke test verifies below --
     * proving the new per-boid/scene-level CCheckpoint fields (design/05_viz_contract.md §2.1/
     * §2.2) are genuinely reachable from real C, not just from murmur_ffi's own Rust tests. */
    const char *hooks[3] = {"boid_state_machine", "ecology", "obstacles"};
    config.step_hooks = hooks;
    config.step_hooks_len = 3;

    struct MurmurSimulation *sim = murmur_create(&config);
    if (sim == NULL) {
        fprintf(stderr, "murmur_create failed: %s\n", murmur_last_error_message());
        return 1;
    }

    if (murmur_boid_count(sim) != 40) {
        fprintf(stderr, "unexpected boid_count\n");
        return 1;
    }

    if (murmur_warning_count(sim) != 1) {
        fprintf(stderr, "expected exactly 1 construction Warning (mismatched cell_size), got %u\n",
                murmur_warning_count(sim));
        return 1;
    }
    const char *warning = murmur_warning_message(sim, 0);
    if (warning == NULL || strstr(warning, "cell_size") == NULL) {
        fprintf(stderr, "warning message missing or doesn't mention cell_size: %s\n",
                warning == NULL ? "(null)" : warning);
        return 1;
    }

    struct CCommand commands[4];
    memset(commands, 0, sizeof(commands));
    commands[0].kind = CMD_SET_CHECKPOINT_STRIDE;
    commands[0].stride = 5;
    /* CMD_SET_ENVIRONMENT's write direction (design/05_viz_contract.md §3): proves it reaches
     * a real composed ecology plugin from real C, not just murmur_ffi's own Rust tests. */
    commands[1].kind = CMD_SET_ENVIRONMENT;
    commands[1].env_day = 100;
    commands[1].env_hour = 10.0;
    /* CMD_ADD_OBSTACLE/CMD_REMOVE_OBSTACLE's write direction: removes the construction-time
     * default sphere (id 0) and adds a box instead, proving both reach a real composed
     * obstacles plugin from real C. */
    commands[2].kind = CMD_REMOVE_OBSTACLE;
    commands[2].id = 0;
    commands[3].kind = CMD_ADD_OBSTACLE;
    commands[3].obstacle_primitive.kind = 1; /* Box */
    commands[3].obstacle_primitive.center.x = 1.0;
    commands[3].obstacle_primitive.center.y = 2.0;
    commands[3].obstacle_primitive.center.z = 3.0;
    commands[3].obstacle_primitive.half_extent.x = 4.0;
    commands[3].obstacle_primitive.half_extent.y = 4.0;
    commands[3].obstacle_primitive.half_extent.z = 4.0;
    commands[3].obstacle_csg_op = 0; /* Union */

    struct MurmurCheckpointBuffer *buffer = NULL;
    int32_t status = murmur_run_batch(sim, 15, 1, commands, 4, &buffer);
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
    /* dt=1.0, hours_per_dt defaults to 0.5 -> 15 steps advance 7.5 hours past the injected
     * day 100, hour 10.0 -> day 100, hour 17.5. */
    if (cp.environment.day != 100 || cp.environment.hour < 17.0 || cp.environment.hour > 18.0) {
        fprintf(stderr, "CMD_SET_ENVIRONMENT didn't reach ecology: day=%llu hour=%f\n",
                (unsigned long long)cp.environment.day, cp.environment.hour);
        return 1;
    }
    if (cp.obstacle_count != 1 || cp.obstacles[0].id != 1 || cp.obstacles[0].primitive.kind != 1) {
        fprintf(stderr, "CMD_ADD_OBSTACLE/CMD_REMOVE_OBSTACLE didn't reach obstacles: "
                        "count=%u id=%u kind=%u\n",
                cp.obstacle_count, cp.obstacle_count ? cp.obstacles[0].id : 0,
                cp.obstacle_count ? cp.obstacles[0].primitive.kind : 0);
        return 1;
    }

    murmur_checkpoint_buffer_destroy(buffer);
    murmur_destroy(sim);

    printf("C_SMOKE_OK\n");
    return 0;
}
