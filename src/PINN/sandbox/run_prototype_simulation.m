function run_prototype_simulation()
% AURA-MFP 4-DOF Digital Twin — interactive demo
%
%   Auto mode  : sweeps every DOF through its full mechanical range
%   Manual mode: pose the panel yourself with the sliders so you can
%                walk an audience through efficiency vs. tilt angle
%   Play/Pause : freeze the sim so you can orbit the camera, then resume

    %% 0. Geometry constants (mirror AURA_MFP_panel_joint.scad) ----------
    G.BASE_THICK     = 10;
    G.LOWER_H        = 220;
    G.UPPER_H        = 160;
    G.UPPER_TRAVEL   = 130;
    G.YAW_THICK      = 16;
    G.ARM_H          = 55;
    G.ARM_BASE_THICK =  8;
    G.PIVOT_BOSS_D   = 12;
    G.PM_THICK = 6;
    G.PM_LUG_H = 14;
    

    % Tunable visual offsets ------------------------------------------------
    G.PANEL_Z        = 70;       % panel face height above roll pivot (mm)
                                 %   -> with h=100 this puts the panel
                                 %      face at absolute Z ≈ 471 mm
                    
    G.YAW_SEAT_Z = G.UPPER_H/2 + G.YAW_THICK/2;   % 88 mm — yaw disk height
                                                  %   above inner-post center.
                                                  %   Decrease to lower the
                                                  %   entire yaw/pitch/roll/
                                                  %   panel assembly.
    % DOF ranges ------------------------------------------------------------
    G.H_MAX     = G.UPPER_TRAVEL;   % 100 mm
    G.YAW_LIM   = 180;              % deg
    G.PITCH_LIM =  35;              % deg
    G.ROLL_LIM  =  25;              % deg

    %% 1. Figure & axes ---------------------------------------------------
    fig = figure('Name','AURA-MFP 4-DOF Digital Twin', ...
                 'Color','w','Position',[80 80 1200 780]);
    ax  = axes('Parent',fig,'Units','pixels','Position',[60 180 900 560]);
    hold(ax,'on'); grid(ax,'on'); axis(ax,'equal'); view(ax,135,20);
    xlabel(ax,'X (mm)'); ylabel(ax,'Y (mm)'); zlabel(ax,'Z (mm)');
    xlim(ax,[-300 300]); ylim(ax,[-300 300]); zlim(ax,[0 700]);
    camlight(ax,'headlight'); lighting(ax,'gouraud');
    rotate3d(ax,'on');     % click-and-drag orbit any time

    %% 2. Load STL meshes -------------------------------------------------
    fprintf('Loading STL meshes...\n');
    here = fileparts(mfilename('fullpath'));
    M.base        = stlread(fullfile(here,'AURA_MFP_base_plate.stl'));
    M.post_outer  = stlread(fullfile(here,'AURA_MFP_post_outer.stl'));
    M.post_inner  = stlread(fullfile(here,'AURA_MFP_post_inner.stl'));
    M.yaw_disk    = stlread(fullfile(here,'AURA_MFP_yaw_disk.stl'));
    M.pitch_b     = stlread(fullfile(here,'AURA_MFP_pitch_bracket.stl'));
    M.roll_b      = stlread(fullfile(here,'AURA_MFP_roll_bracket.stl'));
    M.panel_plate = stlread(fullfile(here,'AURA_MFP_panel_mount_plate.stl'));

    %% 3. Kinematic tree --------------------------------------------------
    % Split each rotating joint into a "rotation node" and a "render node"
    % so that rigid bodies (e.g. pitch_bracket) don't rotate with their
    % downstream joints.
    t_base         = hgtransform('Parent',ax);
    t_post_outer   = hgtransform('Parent',t_base);          % renders outer
    t_post_inner   = hgtransform('Parent',t_post_outer);    % renders inner
    t_yaw          = hgtransform('Parent',t_post_inner);    % renders yaw disk
    t_pitch_base   = hgtransform('Parent',t_yaw);           % renders pitch bracket (static)
    t_pitch_arm    = hgtransform('Parent',t_yaw);           % rotates with pitch
    t_roll_base    = hgtransform('Parent',t_pitch_arm);     % renders roll bracket (flipped up)
    t_roll_arm     = hgtransform('Parent',t_pitch_arm);     % rotates with roll
    t_panel        = hgtransform('Parent',t_roll_arm);      % renders panel plate

    render_part(M.base,        t_base,        [0.28 0.46 0.63]);
    render_part(M.post_outer,  t_post_outer,  [0.69 0.77 0.87]);
    render_part(M.post_inner,  t_post_inner,  [0.39 0.58 0.93]);
    render_part(M.yaw_disk,    t_yaw,         [0.12 0.56 1.00]);
    render_part(M.pitch_b,     t_pitch_base,  [0.25 0.41 0.88]);
    render_part(M.roll_b,      t_roll_base,   [0.00 0.00 0.50]);
    render_part(M.panel_plate, t_panel,       [0.10 0.10 0.44]);

    %% 4. Static offsets --------------------------------------------------
    set(t_post_outer, 'Matrix', ...
        makehgtform('translate',[0 0 G.BASE_THICK + G.LOWER_H/2]));
    
    set(t_pitch_base, 'Matrix', ...
        makehgtform('translate',[0 0 G.YAW_THICK/2]));
    
    
    set(t_roll_base, 'Matrix', makehgtform('xrotate', pi));

    set(t_panel, 'Matrix', ...
        makehgtform('translate',[0 0 G.PM_THICK + G.PM_LUG_H]) * ...
        makehgtform('xrotate', pi));

    %% View preset buttons (east side of figure) --------------------------
    % Each preset = [azimuth, elevation] in degrees, matching MATLAB's view().
    views = {
        'Iso',         [135,  20]
        'Front (+Y)',  [  0,   0]
        'Back  (-Y)',  [180,   0]
        'Right (+X)',  [ 90,   0]
        'Left  (-X)',  [-90,   0]
        'Top',         [  0,  90]
        'Bottom',      [  0, -90]
        'Panel face',  [  0,  15]   % audience POV looking at the panel
        'Operator',    [ 35,  10]   % over-the-shoulder demo angle
    };
    
    btn_w   = 110;
    btn_h   = 30;
    btn_gap = 6;
    fig_pos = get(fig,'Position');
    x0      = fig_pos(3) - btn_w - 15;             % 15 px from right edge
    y_top   = fig_pos(4) - 40;                     % start near top
    panel_h = numel(views)*(btn_h+btn_gap) + 50;
    
    uicontrol('Parent',fig,'Style','text', ...
        'String','Camera presets', ...
        'Position',[x0 y_top btn_w 20], ...
        'FontWeight','bold','BackgroundColor','w', ...
        'HorizontalAlignment','center');
    
    for k = 1:size(views,1)
        name = views{k,1};
        az_el = views{k,2};
        y_k = y_top - k*(btn_h+btn_gap);
        uicontrol('Parent',fig,'Style','pushbutton', ...
            'String',name, ...
            'Position',[x0 y_k btn_w btn_h], ...
            'Callback',@(~,~) set_view(ax, az_el));
    end

    %% 5. UI controls -----------------------------------------------------
    state = struct('playing',true,'manual',false,'fig',fig);
    setappdata(fig,'state',state);

    % --- Play/Pause -----------------------------------------------------
    btn_play = uicontrol('Parent',fig,'Style','pushbutton', ...
        'String','Pause','Position',[20 130 100 32], ...
        'FontWeight','bold','Callback',@toggle_play);

    % --- Auto / Manual --------------------------------------------------
    btn_mode = uicontrol('Parent',fig,'Style','togglebutton', ...
        'String','Auto sweep','Position',[20 90 100 32], ...
        'Callback',@toggle_mode);

    % --- Reset ----------------------------------------------------------
    uicontrol('Parent',fig,'Style','pushbutton', ...
        'String','Reset','Position',[20 50 100 32], ...
        'Callback',@(~,~) reset_sliders());

    % --- Stop -----------------------------------------------------------
    uicontrol('Parent',fig,'Style','pushbutton', ...
        'String','Stop Sim','Position',[20 10 100 32], ...
        'Callback',@(~,~) setappdata(fig,'run_flag',false));

    % --- Sliders --------------------------------------------------------
    sliders = struct();
    sliders.height = make_slider(fig, 'Height (mm)',   [150  90], 0, G.H_MAX,     50);
    sliders.yaw    = make_slider(fig, 'Yaw   (°)',     [430  90], -G.YAW_LIM,   G.YAW_LIM,   0);
    sliders.pitch  = make_slider(fig, 'Pitch (°)',     [710  90], -G.PITCH_LIM, G.PITCH_LIM, 0);
    sliders.roll   = make_slider(fig, 'Roll  (°)',     [990  90], -G.ROLL_LIM,  G.ROLL_LIM,  0);

    setappdata(fig,'sliders',sliders);
    setappdata(fig,'run_flag',true);

    %% 6. Main loop -------------------------------------------------------
    fprintf('Starting digital twin synchronization...\n');
    fprintf('  Pause to freeze. Toggle "Auto sweep" -> "Manual" to pose.\n');
    t_anim = 0;
    last_tic = tic;
    while ishandle(fig) && getappdata(fig,'run_flag')
        st = getappdata(fig,'state');

        % Advance demo clock only when playing AND in auto mode
        dt = toc(last_tic); last_tic = tic;
        if st.playing && ~st.manual
            t_anim = t_anim + dt;
        end

        % --- Determine commanded pose -------------------------------------
        if st.manual
            h = sliders.height.get();
            y = sliders.yaw.get();
            p = sliders.pitch.get();
            r = sliders.roll.get();
        else
            h = 50 + 50 * sin(2*pi*t_anim/12);
            y = G.YAW_LIM   * sin(2*pi*t_anim/8);
            p = G.PITCH_LIM * sin(2*pi*t_anim/5);
            r = G.ROLL_LIM  * cos(2*pi*t_anim/4);
            % Echo into the sliders so the audience can read them
            sliders.height.set(h);
            sliders.yaw.set(y);
            sliders.pitch.set(p);
            sliders.roll.set(r);
        end

        % --- Apply kinematics ---------------------------------------------
        set(t_post_inner,'Matrix', makehgtform('translate',[0 0 43.8 + h]));
        set(t_yaw,       'Matrix', ...
            makehgtform('translate',[0 0 G.YAW_SEAT_Z]) * ...
            makehgtform('zrotate', deg2rad(y)));
        set(t_pitch_arm, 'Matrix', ...
            makehgtform('translate',[0 0 G.ARM_BASE_THICK + G.ARM_H - G.PIVOT_BOSS_D/2]) * ...
            makehgtform('xrotate', deg2rad(p)));
        set(t_roll_arm, 'Matrix', ...
            makehgtform('translate',[0 0 G.ARM_H - G.PIVOT_BOSS_D/2]) * ...
            makehgtform('yrotate', deg2rad(r)));

        % --- HUD ----------------------------------------------------------
        mode_str = ternary(st.manual,'MANUAL','AUTO');
        play_str = ternary(st.playing,'▶','‖');
        title(ax, sprintf(['%s  %s   |   h=%5.1f mm   yaw=%+6.1f°   ' ...
                           'pitch=%+5.1f°   roll=%+5.1f°'], ...
            play_str, mode_str, h, y, p, r), ...
            'FontWeight','bold','FontName','Consolas');

        drawnow limitrate;
        if ~st.playing
            % Idle while paused so CPU stays cool but UI stays responsive
            pause(0.03);
        end
    end
    fprintf('Simulation stopped.\n');

    %% ===== nested helpers ==============================================
    function toggle_play(src,~)
        s = getappdata(fig,'state');
        s.playing = ~s.playing;
        setappdata(fig,'state',s);
        src.String = ternary(s.playing,'Pause','Play');
        last_tic = tic;   % don't accumulate dt while paused
    end

    function toggle_mode(src,~)
        s = getappdata(fig,'state');
        s.manual = logical(src.Value);
        setappdata(fig,'state',s);
        src.String = ternary(s.manual,'Manual','Auto sweep');
    end

    function reset_sliders()
        sliders.height.set(0);
        sliders.yaw.set(0);
        sliders.pitch.set(0);
        sliders.roll.set(0);
    end
end

%% ========================================================================
function render_part(mesh_data, parent_handle, color_vec)
    patch('Faces',mesh_data.ConnectivityList, ...
          'Vertices',mesh_data.Points, ...
          'Parent',parent_handle, ...
          'FaceColor',color_vec, ...
          'EdgeColor','none', ...
          'FaceLighting','gouraud', ...
          'AmbientStrength',0.4, ...
          'DiffuseStrength',0.6, ...
          'SpecularStrength',0.2);
end

function s = make_slider(parent, label, pos, lo, hi, init)
% Build a labelled slider with a numeric readout.
% Returns a struct with get() and set() handles.
    x = pos(1); y = pos(2);
    uicontrol('Parent',parent,'Style','text','String',label, ...
        'Position',[x y+38 220 18],'HorizontalAlignment','left', ...
        'FontWeight','bold','BackgroundColor','w');
    sld = uicontrol('Parent',parent,'Style','slider', ...
        'Min',lo,'Max',hi,'Value',init, ...
        'SliderStep',[0.01 0.1], ...
        'Position',[x y 220 22]);
    txt = uicontrol('Parent',parent,'Style','edit', ...
        'String',sprintf('%.1f',init), ...
        'Position',[x+150 y+22 70 18], ...
        'BackgroundColor','w');

    addlistener(sld,'Value','PostSet', @(~,~) set(txt,'String', ...
        sprintf('%.1f', get(sld,'Value'))));
    set(txt,'Callback',@(src,~) clamp_and_set(src,sld,lo,hi));

    s.get = @() get(sld,'Value');
    s.set = @(v) setval(sld,txt,max(lo,min(hi,v)));
end

function setval(sld,txt,v)
    set(sld,'Value',v);
    set(txt,'String',sprintf('%.1f',v));
end

function clamp_and_set(src,sld,lo,hi)
    v = str2double(get(src,'String'));
    if isnan(v), v = get(sld,'Value'); end
    v = max(lo,min(hi,v));
    set(sld,'Value',v);
    set(src,'String',sprintf('%.1f',v));
end

function out = ternary(cond,a,b)
    if cond, out = a; else, out = b; end
end

function set_view(ax, az_el)
    view(ax, az_el(1), az_el(2));
    camva(ax,'auto');     % reset zoom so the preset framing looks consistent
    drawnow;
end
    
