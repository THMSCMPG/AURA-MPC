// ============================================================================
// AURA-MFP Panel Joint Assembly — Parametric 4-DOF Manual Mount
// W. T. Campagna, 2026
//
// Parts (print separately):
//   1. base_plate()        — weighted base with square post socket
//   2. post_outer()        — fixed lower post tube (square section)
//   3. post_inner()        — sliding upper post (fits inside outer)
//   4. post_pin_collar()   — locking collar with pin holes (height lock)
//   5. yaw_disk()          — rotating yaw platform with friction clamp
//   6. pitch_bracket()     — U-bracket for pitch pivot (forward/back tilt)
//   7. roll_bracket()      — U-bracket for roll pivot (side tilt), 90° to pitch
//   8. panel_mount_plate() — flat plate that attaches to panel frame
//
// Hardware needed (all M5 or M8 hex bolts + wing nuts):
//   - 4× M5×30 bolt + wing nut  (pitch pivot + roll pivot)
//   - 1× M5×20 bolt + wing nut  (yaw clamp)
//   - 1× 6mm dia × 40mm steel pin OR M6 bolt (height lock through post)
//   - 8× M5×16 flat head screws (panel mount plate to panel frame)
//   - Rubber feet (4×) under base plate
//
// Print notes:
//   PLA:   min 40% gyroid infill, 4 perimeters, 0.2mm layer
//          ADD: outdoor UV/heat coating (Rustoleum or polyurethane)
//          WARNING: PLA softens ~60C — adequate for brief field sessions
//          but not for unattended deployment in direct summer sun
//   PETG:  preferred for outdoor use, same settings
//   Resin: 2mm walls OK, print solid for pivot areas, sand pin holes
//
// Coordinate system: Z = up, panel faces +Y direction
// ============================================================================

// ── Global parameters ────────────────────────────────────────────────────────

// Post (square cross-section, prevents unwanted yaw during height adjustment)
POST_OD       = 28;    // outer post outer side (mm) — square
POST_WALL     = 3;     // wall thickness
POST_ID       = POST_OD - 2*POST_WALL;  // inner clear dimension

// Lower (fixed) post
LOWER_H       = 220;   // height of lower post section

// Upper (sliding) post
UPPER_H       = 160;   // length of upper sliding section
UPPER_TRAVEL  = 100;   // max usable travel
PIN_HOLE_D    = 6.5;   // through-hole for M6 height-lock pin
PIN_SPACING   = 15;    // distance between pin holes on inner post
CLEARANCE     = 0.4;   // sliding fit clearance

// Base plate
BASE_W        = 160;
BASE_H        = 160;
BASE_THICK    = 10;
BASE_FOOT_D   = 12;    // rubber foot recess diameter
BASE_FOOT_H   = 3;     // rubber foot recess depth
SOCKET_D      = POST_OD + 4;  // extra material around post socket

// Yaw disk
YAW_D         = 90;    // outer diameter of yaw disk
YAW_THICK     = 16;
YAW_CLAMP_W   = 15;    // width of clamp slot
YAW_BOLT_D    = 5.5;   // M5 clamp bolt hole

// Pivot arms (pitch and roll brackets)
ARM_H         = 55;    // height of U-bracket arms
ARM_W         = 18;    // arm wall thickness
ARM_SPAN      = 50;    // clear span between arms (panel edge fits here)
PIVOT_BOLT_D  = 5.5;   // M5 pivot bolt
PIVOT_BOSS_D  = 12;    // boss OD around pivot hole

// Panel mount plate
PM_W          = 380;   // width (adjust to your panel frame width)
PM_H          = 240;   // height (adjust to your panel frame height)
PM_THICK      = 6;
PM_HOLE_D     = 5.5;   // M5 mounting holes
PM_HOLE_IN    = 15;    // inset from corners

// ── Utility modules ──────────────────────────────────────────────────────────

module rounded_rect(w, h, d, r=5) {
    // 2D rounded rectangle centered at origin
    offset(r=r) square([w-2*r, h-2*r], center=true);
}

module square_tube(od, wall, h) {
    difference() {
        cube([od, od, h], center=true);
        cube([od-2*wall, od-2*wall, h+1], center=true);
    }
}

module bolt_hole(d, h, head_d=0, head_h=0) {
    cylinder(d=d, h=h, center=true, $fn=20);
    if (head_d > 0)
        translate([0, 0, h/2 - head_h])
            cylinder(d=head_d, h=head_h+0.1, $fn=20);
}

// ── 1. Base Plate ─────────────────────────────────────────────────────────────
module base_plate() {
    difference() {
        union() {
            // Main plate
            linear_extrude(BASE_THICK)
                rounded_rect(BASE_W, BASE_H, BASE_THICK, r=8);
            // Post socket boss (extra material for rigidity)
            translate([0, 0, BASE_THICK])
                linear_extrude(BASE_THICK * 1.5)
                    rounded_rect(SOCKET_D+8, SOCKET_D+8, 4, r=4);
        }
        // Square post socket (through base + boss)
        translate([0, 0, -1])
            cube([POST_OD + CLEARANCE, POST_OD + CLEARANCE,
                  BASE_THICK*2.5 + 2], center=true);
        // Post locking bolt hole (horizontal, through socket boss)
        translate([0, 0, BASE_THICK + BASE_THICK*0.75])
            rotate([0, 90, 0])
                cylinder(d=YAW_BOLT_D, h=SOCKET_D+12, center=true, $fn=20);
        // Rubber foot recesses (4 corners)
        for (dx=[-1,1]) for (dy=[-1,1])
            translate([dx*(BASE_W/2-BASE_FOOT_D), dy*(BASE_H/2-BASE_FOOT_D), -1])
                cylinder(d=BASE_FOOT_D, h=BASE_FOOT_H+1, $fn=20);
        // Weight pockets (reduce PLA mass, can be filled with sand/steel shot)
        for (dx=[-1,1]) for (dy=[-1,1])
            translate([dx*BASE_W/4, dy*BASE_H/4, BASE_THICK*0.4])
                cylinder(d=28, h=BASE_THICK*0.6+1, $fn=30);
    }
}

// ── 2. Lower Post (outer tube, fixed) ────────────────────────────────────────
module post_outer() {
    difference() {
        // Square tube
        square_tube(POST_OD, POST_WALL, LOWER_H);
        // Pin hole at top (for positional reference / stop)
        translate([0, 0, LOWER_H/2 - 10])
            rotate([0, 90, 0])
                cylinder(d=PIN_HOLE_D, h=POST_OD+2, center=true, $fn=20);
        // Cable routing slot on one face (20mm wide, starts 15mm from bottom)
        translate([POST_OD/2 - POST_WALL/2, 0, -LOWER_H/2 + 15 + 60])
            cube([POST_WALL+1, 12, LOWER_H - 80], center=true);
    }
}

// ── 3. Inner/Upper Post (sliding tube) ───────────────────────────────────────
module post_inner() {
    id_fit = POST_ID - CLEARANCE;
    difference() {
        union() {
            // Square solid section (fits inside outer)
            cube([id_fit, id_fit, UPPER_H], center=true);
            // Top flange for yaw disk mounting
            translate([0, 0, UPPER_H/2 - 5])
                cube([POST_OD + 6, POST_OD + 6, 10], center=true);
        }
        // Hollow center (cable routing)
        cube([id_fit - 2*POST_WALL, id_fit - 2*POST_WALL, UPPER_H+2], center=true);
        // Height-lock pin holes (every PIN_SPACING mm along lower 3/4)
        for (i=[0:floor(UPPER_TRAVEL/PIN_SPACING)])
            translate([0, 0, -UPPER_H/2 + 20 + i*PIN_SPACING])
                rotate([0, 90, 0])
                    cylinder(d=PIN_HOLE_D, h=id_fit+2, center=true, $fn=20);
        // Yaw disk mount holes in top flange (4× M5)
        for (dx=[-1,1]) for (dy=[-1,1])
            translate([dx*15, dy*15, UPPER_H/2 - 8])
                cylinder(d=5.5, h=12, center=true, $fn=20);
    }
}

// ── 4. Post Pin Collar ────────────────────────────────────────────────────────
// Slides over outer post, aligns pin holes
module post_pin_collar() {
    collar_h = 20;
    difference() {
        cylinder(d=POST_OD + 12, h=collar_h, center=true, $fn=40);
        // Inner square (over outer post)
        cube([POST_OD + 0.4, POST_OD + 0.4, collar_h + 2], center=true);
        // Pin hole (aligned)
        rotate([0, 90, 0])
            cylinder(d=PIN_HOLE_D, h=POST_OD+16, center=true, $fn=20);
    }
}

// ── 5. Yaw Disk ───────────────────────────────────────────────────────────────
// Sits on top of inner post, can rotate freely until clamped
module yaw_disk() {
    difference() {
        union() {
            // Main disk
            cylinder(d=YAW_D, h=YAW_THICK, center=true, $fn=60);
            // Raised platform for pitch bracket mounting
            translate([0, 0, YAW_THICK/2])
                cube([40, 40, 8], center=true);
        }
        // Center hole (inner post top passes through)
        cylinder(d=POST_OD + CLEARANCE*2, h=YAW_THICK+2, center=true, $fn=40);
        // Clamp slot (radial, for M5 friction lock)
        translate([YAW_D/4, 0, 0])
            cube([YAW_D/2 + 2, YAW_CLAMP_W, YAW_THICK + 2], center=true);
        // Clamp bolt hole (tangential through slot)
        rotate([0, 90, 90])
            cylinder(d=YAW_BOLT_D, h=YAW_D + 2, center=true, $fn=20);
        // Degree markings (shallow grooves every 15°) — decorative / functional
        for (a=[0:15:345])
            rotate([0, 0, a])
                translate([YAW_D/2 - 4, 0, YAW_THICK/2 - 1])
                    cube([4, 1, 2], center=true);
        // Bolt holes to mount on inner post flange (4× M5)
        for (dx=[-1,1]) for (dy=[-1,1])
            translate([dx*15, dy*15, 0])
                cylinder(d=5.5, h=YAW_THICK+2, center=true, $fn=20);
    }
}

// ── 6. Pitch Bracket ──────────────────────────────────────────────────────────
// U-bracket on top of yaw disk, pivot axis = X (forward/back tilt)
module pitch_bracket() {
    base_thick = 8;
    difference() {
        union() {
            // Base plate (mounts to yaw disk)
            translate([0, 0, base_thick/2])
                cube([ARM_SPAN + 2*ARM_W, 50, base_thick], center=true);
            // Two vertical arms
            for (sx=[-1,1])
                translate([sx*(ARM_SPAN/2 + ARM_W/2), 0, base_thick + ARM_H/2])
                    cube([ARM_W, 40, ARM_H], center=true);
            // Pivot boss on each arm
            for (sx=[-1,1])
                translate([sx*(ARM_SPAN/2 + ARM_W/2), 0, base_thick + ARM_H - PIVOT_BOSS_D/2])
                    rotate([90, 0, 0])
                        cylinder(d=PIVOT_BOSS_D, h=44, center=true, $fn=30);
        }
        // Pivot bolt holes through arms (M5, horizontal)
        for (sx=[-1,1])
            translate([sx*(ARM_SPAN/2 + ARM_W/2), 0, base_thick + ARM_H - PIVOT_BOSS_D/2])
                rotate([90, 0, 0])
                    cylinder(d=PIVOT_BOLT_D, h=ARM_SPAN + 2*ARM_W + 2, center=true, $fn=20);
        // Wing nut access (countersink on outer face)
        for (sx=[-1,1])
            translate([sx*(ARM_SPAN/2 + ARM_W - 1), 0, base_thick + ARM_H - PIVOT_BOSS_D/2])
                rotate([90, 0, 0])
                    cylinder(d=12, h=8, center=true, $fn=30);
        // Base mount holes (4× M5 to yaw disk)
        for (dx=[-1,1]) for (dy=[-1,1])
            translate([dx*15, dy*15, 0])
                cylinder(d=5.5, h=base_thick+2, center=true, $fn=20);
        // Pitch angle marks (on arm face, ±35° = PINN pitch limit)
        for (a=[-35,-20,-10,0,10,20,35])
            translate([ARM_SPAN/2 + ARM_W, 
                       (ARM_H - PIVOT_BOSS_D/2)*sin(a),
                       base_thick + (ARM_H - PIVOT_BOSS_D/2)*cos(a)])
                rotate([0, 0, 0])
                    cube([ARM_W+1, 2, 1.5], center=true);
    }
}

// ── 7. Roll Bracket ───────────────────────────────────────────────────────────
// Connects pitch pivot to panel mount, pivot axis = Y (side tilt)
// This piece is the "inner" yoke — it hangs on the pitch bracket pivot pin
module roll_bracket() {
    yoke_thick = 8;
    difference() {
        union() {
            // Yoke body (rides on pitch pivot)
            translate([0, 0, 0])
                cube([ARM_SPAN - CLEARANCE*2, yoke_thick*3, yoke_thick], center=true);
            // Two arms going "down" (toward panel) for roll pivot
            for (sy=[-1,1])
                translate([0, sy*(ARM_SPAN/2 + ARM_W/2 - yoke_thick), -(ARM_H/2)])
                    cube([45, ARM_W, ARM_H], center=true);
            // Roll pivot boss on each arm
            for (sy=[-1,1])
                translate([0, sy*(ARM_SPAN/2 + ARM_W/2 - yoke_thick), -(ARM_H - PIVOT_BOSS_D/2)])
                    rotate([0, 90, 0])
                        cylinder(d=PIVOT_BOSS_D, h=49, center=true, $fn=30);
        }
        // Pitch pivot hole (horizontal, through yoke)
        rotate([90, 0, 0])
            cylinder(d=PIVOT_BOLT_D, h=ARM_SPAN + 2, center=true, $fn=20);
        // Roll pivot holes (perpendicular to pitch)
        for (sy=[-1,1])
            translate([0, sy*(ARM_SPAN/2 + ARM_W/2 - yoke_thick), -(ARM_H - PIVOT_BOSS_D/2)])
                rotate([0, 90, 0])
                    cylinder(d=PIVOT_BOLT_D, h=50+2, center=true, $fn=20);
        // Wing nut countersink
        for (sy=[-1,1])
            translate([25, sy*(ARM_SPAN/2 + ARM_W/2 - yoke_thick), -(ARM_H - PIVOT_BOSS_D/2)])
                rotate([0, 90, 0])
                    cylinder(d=12, h=8, center=true, $fn=30);
        // Roll angle marks (±25°)
        for (a=[-25,-15,-5,0,5,15,25])
            translate([(ARM_SPAN/2 - 2)*sin(a),
                       ARM_SPAN/2 + ARM_W/2,
                       -(ARM_H - PIVOT_BOSS_D/2 - (ARM_SPAN/2 - 2)*cos(a))])
                cube([2, ARM_W+1, 1.5], center=true);
    }
}

// ── 8. Panel Mount Plate ──────────────────────────────────────────────────────
// Flat plate that attaches to the panel frame AND connects to roll bracket pivot
module panel_mount_plate() {
    difference() {
        union() {
            // Main plate
            linear_extrude(PM_THICK)
                rounded_rect(PM_W, PM_H, PM_THICK, r=10);
            // Roll pivot lugs (two ears extending from center back edge)
            for (sx=[-1,1])
                translate([sx*(ARM_SPAN/2 + ARM_W/2), 0, PM_THICK/2])
                    cube([ARM_W, 35, PM_THICK + 15], center=true);
            // Pivot boss on each lug
            for (sx=[-1,1])
                translate([sx*(ARM_SPAN/2 + ARM_W/2), 0, PM_THICK + 14])
                    rotate([90, 0, 0])
                        cylinder(d=PIVOT_BOSS_D, h=39, center=true, $fn=30);
        }
        // Roll pivot holes through lugs
        for (sx=[-1,1])
            translate([sx*(ARM_SPAN/2 + ARM_W/2), 0, PM_THICK + 14])
                rotate([90, 0, 0])
                    cylinder(d=PIVOT_BOLT_D, h=40, center=true, $fn=20);
        // Panel frame mounting holes (4 corners + midpoints)
        for (dx=[-1,1]) for (dy=[-1,1])
            translate([dx*(PM_W/2 - PM_HOLE_IN), dy*(PM_H/2 - PM_HOLE_IN), 0])
                cylinder(d=PM_HOLE_D, h=PM_THICK+2, center=true, $fn=20);
        for (dx=[-1,1])
            translate([dx*(PM_W/4), 0, 0])
                cylinder(d=PM_HOLE_D, h=PM_THICK+2, center=true, $fn=20);
        // Weight relief pockets
        for (dx=[-1,1]) for (dy=[-1,1])
            translate([dx*PM_W/4, dy*PM_H/4, PM_THICK*0.3])
                linear_extrude(PM_THICK*0.7+1)
                    rounded_rect(PM_W/4, PM_H/4, 3, r=8);
        // Sensor cable pass-throughs (center)
        translate([0, 0, 0])
            cylinder(d=20, h=PM_THICK+2, center=true, $fn=30);
    }
}

// ── Assembly preview (comment out individual parts for printing) ─────────────

module full_assembly() {
    // Base
    color("SteelBlue", 0.9) base_plate();
    // Lower post (sitting in base socket)
    color("LightSteelBlue", 0.9)
        translate([0, 0, BASE_THICK])
            post_outer();
    // Inner post (extending out of outer post)
    color("CornflowerBlue", 0.85)
        translate([0, 0, BASE_THICK + LOWER_H - UPPER_H + 60])
            post_inner();
    // Yaw disk
    color("DodgerBlue", 0.85)
        translate([0, 0, BASE_THICK + LOWER_H + 5])
            yaw_disk();
    // Pitch bracket
    color("RoyalBlue", 0.85)
        translate([0, 0, BASE_THICK + LOWER_H + 5 + YAW_THICK/2 + 4])
            pitch_bracket();
    // Roll bracket + panel plate
    color("Navy", 0.8)
        translate([0, 0, BASE_THICK + LOWER_H + 5 + YAW_THICK/2 + 4
                   + 8 + ARM_H + 10])
            roll_bracket();
    color("MidnightBlue", 0.75)
        translate([0, 0, BASE_THICK + LOWER_H + 5 + YAW_THICK/2 + 4
                   + 8 + ARM_H + 10 + ARM_H + 20])
            panel_mount_plate();
}

// ── Render mode (uncomment the part to export) ───────────────────────────────

full_assembly();  // comment out for print mode

// To print individually, comment full_assembly() and uncomment one:
// base_plate();
// post_outer();
// post_inner();
// post_pin_collar();
// yaw_disk();
// pitch_bracket();
// roll_bracket();
// panel_mount_plate();
