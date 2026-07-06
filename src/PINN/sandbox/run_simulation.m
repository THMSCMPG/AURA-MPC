function run_simulation()
% AURA-MPC closed-loop simulator with live PINN vs RK4TRAN inspection.

    here = fileparts(mfilename('fullpath'));
    pinnRoot = fileparts(here);
    configPath = fullfile(pinnRoot, 'configs', 'sandbox.yaml');
    checkpointPath = fullfile(pinnRoot, 'outputs', 'pretrain', 'checkpoints', 'best_model.pt');
    outputDir = fullfile(pinnRoot, 'outputs', 'simulation');
    requirementsPath = fullfile(pinnRoot, 'requirements.txt');
    venvDir = fullfile(pinnRoot, '.matlab-venv');

    fig = figure( ...
        'Name', 'AURA-MPC Closed-Loop Simulator', ...
        'Color', 'w', ...
        'Position', [40 40 1500 900], ...
        'CloseRequestFcn', @onClose);

    axTwin = axes('Parent', fig, 'Units', 'pixels', 'Position', [40 340 860 520]);
    hold(axTwin, 'on');
    grid(axTwin, 'on');
    axis(axTwin, 'equal');
    view(axTwin, 135, 20);
    xlabel(axTwin, 'X (mm)');
    ylabel(axTwin, 'Y (mm)');
    zlabel(axTwin, 'Z (mm)');
    xlim(axTwin, [-300 300]);
    ylim(axTwin, [-300 300]);
    zlim(axTwin, [0 700]);
    camlight(axTwin, 'headlight');
    lighting(axTwin, 'gouraud');
    rotate3d(axTwin, 'on');

    axReward = axes('Parent', fig, 'Units', 'pixels', 'Position', [40 165 420 130]);
    title(axReward, 'Reward history');
    xlabel(axReward, 'Step');
    ylabel(axReward, 'Reward');
    grid(axReward, 'on');
    hold(axReward, 'on');

    axDrift = axes('Parent', fig, 'Units', 'pixels', 'Position', [480 165 420 130]);
    title(axDrift, 'PINN vs RK4TRAN drift');
    xlabel(axDrift, 'Step');
    yyaxis(axDrift, 'left'); ylabel(axDrift, '\DeltaT (K)');
    yyaxis(axDrift, 'right'); ylabel(axDrift, '\Delta\eta');
    grid(axDrift, 'on');
    hold(axDrift, 'on');

    axPose = axes('Parent', fig, 'Units', 'pixels', 'Position', [40 20 860 120]);
    title(axPose, 'Pose history');
    xlabel(axPose, 'Step');
    ylabel(axPose, 'Degrees / meters');
    grid(axPose, 'on');
    hold(axPose, 'on');

    panelControls = uipanel('Parent', fig, 'Title', 'Controls', 'Units', 'pixels', 'Position', [940 740 520 120]);
    btnStart = uicontrol(panelControls, 'Style', 'pushbutton', 'String', 'Start', ...
        'Position', [15 55 90 35], 'FontWeight', 'bold', 'Callback', @onStart);
    btnPause = uicontrol(panelControls, 'Style', 'pushbutton', 'String', 'Pause', ...
        'Position', [115 55 90 35], 'Callback', @onPause);
    btnReset = uicontrol(panelControls, 'Style', 'pushbutton', 'String', 'Reset', ...
        'Position', [215 55 90 35], 'Callback', @onReset);
    btnStep = uicontrol(panelControls, 'Style', 'pushbutton', 'String', 'Step', ...
        'Position', [315 55 90 35], 'Callback', @onStep);
    chkStepThrough = uicontrol(panelControls, 'Style', 'checkbox', 'String', 'Step-through mode', ...
        'Position', [15 20 150 25], 'Value', 0);
    chkLearning = uicontrol(panelControls, 'Style', 'checkbox', 'String', 'Enable policy learning', ...
        'Position', [185 20 160 25], 'Value', 1);
    txtStatus = uicontrol(panelControls, 'Style', 'text', 'String', 'Idle', ...
        'Position', [360 18 150 28], 'HorizontalAlignment', 'left', 'BackgroundColor', 'w', ...
        'FontWeight', 'bold');

    panelInputs = uipanel('Parent', fig, 'Title', 'Initial conditions', 'Units', 'pixels', 'Position', [940 380 520 350]);
    inputs = struct();
    inputs.lat = make_edit(panelInputs, 'Latitude', 15, 290, '36.5');
    inputs.lon = make_edit(panelInputs, 'Longitude', 175, 290, '-87.3');
    inputs.alt = make_edit(panelInputs, 'Elevation (m)', 335, 290, '100');
    inputs.day = make_edit(panelInputs, 'Day of year', 15, 245, '172');
    inputs.month = make_edit(panelInputs, 'Month', 175, 245, '6');
    inputs.year = make_edit(panelInputs, 'Year', 335, 245, '2024');
    inputs.hour = make_edit(panelInputs, 'Hour', 15, 200, '12');
    inputs.minute = make_edit(panelInputs, 'Minute', 175, 200, '0');
    inputs.ambient = make_edit(panelInputs, 'Ambient C', 335, 200, '25');
    inputs.wind = make_edit(panelInputs, 'Wind (m/s)', 15, 155, '4');
    inputs.winddir = make_edit(panelInputs, 'Wind dir', 175, 155, '180');
    inputs.humidity = make_edit(panelInputs, 'Humidity', 335, 155, '0.5');
    inputs.irradiance = make_edit(panelInputs, 'Irradiance', 15, 110, '850');
    inputs.cloud = make_edit(panelInputs, 'Cloud cover', 175, 110, '0.1');
    inputs.pressure = make_edit(panelInputs, 'Pressure (Pa)', 335, 110, '101325');
    inputs.pitch = make_edit(panelInputs, 'Pitch (deg)', 15, 65, '0');
    inputs.yaw = make_edit(panelInputs, 'Yaw (deg)', 175, 65, '0');
    inputs.roll = make_edit(panelInputs, 'Roll (deg)', 335, 65, '0');
    inputs.z = make_edit(panelInputs, 'Height z (m)', 15, 20, '1.25');
    inputs.period = make_edit(panelInputs, 'Update period (s)', 175, 20, '0.5');

    panelSnapshot = uipanel('Parent', fig, 'Title', 'Live snapshot', 'Units', 'pixels', 'Position', [940 20 520 350]);
    txtPose = make_output(panelSnapshot, 'Current pose', [15 220 240 105]);
    txtPred = make_output(panelSnapshot, 'PINN vs RK4TRAN', [270 220 235 105]);
    txtReward = make_output(panelSnapshot, 'Reward breakdown', [15 95 240 115]);
    txtDecision = make_output(panelSnapshot, 'Decision trace', [270 15 235 195]);
    txtDrift = make_output(panelSnapshot, 'Drift / trust', [15 15 240 70]);

    [geom, twin] = build_twin(axTwin, here);

    app = struct();
    app.bridge = [];
    app.timer = timer( ...
        'ExecutionMode', 'fixedSpacing', ...
        'Period', 0.5, ...
        'BusyMode', 'drop', ...
        'TimerFcn', @onTick);
    app.history = empty_history();
    app.axTwin = axTwin;
    app.axReward = axReward;
    app.axDrift = axDrift;
    app.axPose = axPose;
    app.geom = geom;
    app.twin = twin;
    app.inputs = inputs;
    app.chkStepThrough = chkStepThrough;
    app.chkLearning = chkLearning;
    app.txtStatus = txtStatus;
    app.txtPose = txtPose;
    app.txtPred = txtPred;
    app.txtReward = txtReward;
    app.txtDecision = txtDecision;
    app.txtDrift = txtDrift;
    app.configPath = configPath;
    app.checkpointPath = checkpointPath;
    app.outputDir = outputDir;
    app.requirementsPath = requirementsPath;
    app.venvDir = venvDir;
    app.pythonReady = false;
    guidata(fig, app);

    initialize_runtime();

    function initialize_runtime()
        app = guidata(fig);
        set_status('Preparing Python environment...');
        stop_timer_if_running(app.timer);
        try
            app.bridge = create_bridge( ...
                app.configPath, ...
                app.checkpointPath, ...
                app.outputDir, ...
                app.requirementsPath, ...
                app.venvDir);
            app.pythonReady = true;
        catch err
            app.bridge = [];
            app.pythonReady = false;
            guidata(fig, app);
            set_status('Python setup failed');
            errordlg(sprintf('run_simulation could not prepare its Python runtime.\n\n%s', err.message), ...
                'AURA-MPC Python setup failed', 'modal');
            return;
        end
        guidata(fig, app);
        perform_reset(true);
    end

    function onStart(~, ~)
        app = guidata(fig);
        if ~ensure_bridge_ready()
            return;
        end
        app.timer.Period = max(0.05, str2double(app.inputs.period.value.String));
        if app.chkStepThrough.Value
            set_status('Step-through ready');
            return;
        end
        if strcmp(app.timer.Running, 'off')
            start(app.timer);
        end
        set_status('Running');
    end

    function onPause(~, ~)
        app = guidata(fig);
        stop_timer_if_running(app.timer);
        set_status('Paused');
    end

    function onReset(~, ~)
        if ~ensure_bridge_ready()
            return;
        end
        perform_reset(true);
    end

    function onStep(~, ~)
        if ~ensure_bridge_ready()
            return;
        end
        stop_timer_if_running(guidata(fig).timer);
        advance_once();
        set_status('Stepped');
    end

    function onTick(~, ~)
        if ~ensure_bridge_ready(false)
            return;
        end
        advance_once();
    end

    function perform_reset(clearPlots)
        app = guidata(fig);
        if isempty(app.bridge)
            return;
        end
        stop_timer_if_running(app.timer);
        conditions = gather_conditions();
        pose = gather_pose();
        raw = char(app.bridge.reset(jsonencode(conditions), jsonencode(pose)));
        snapshot = jsondecode(raw);
        if clearPlots
            app.history = empty_history();
            cla(app.axReward);
            cla(app.axDrift);
            cla(app.axPose);
            title(app.axReward, 'Reward history');
            title(app.axDrift, 'PINN vs RK4TRAN drift');
            title(app.axPose, 'Pose history');
            xlabel(app.axReward, 'Step'); ylabel(app.axReward, 'Reward'); grid(app.axReward, 'on'); hold(app.axReward, 'on');
            xlabel(app.axDrift, 'Step'); 
            yyaxis(app.axDrift, 'left'); ylabel(app.axDrift, '\DeltaT (K)');
            yyaxis(app.axDrift, 'right'); ylabel(app.axDrift, '\Delta\eta');
            grid(app.axDrift, 'on'); hold(app.axDrift, 'on');
            xlabel(app.axPose, 'Step'); ylabel(app.axPose, 'Degrees / meters'); grid(app.axPose, 'on'); hold(app.axPose, 'on');
        end
        guidata(fig, app);
        update_snapshot(snapshot, true);
        set_status('Reset');
    end

    function advance_once()
        if ~ishandle(fig)
            return;
        end
        app = guidata(fig);
        if isempty(app.bridge)
            set_status('Python bridge unavailable');
            stop_timer_if_running(app.timer);
            return;
        end
        raw = char(app.bridge.step('', 'mean', logical(app.chkLearning.Value)));
        snapshot = jsondecode(raw);
        update_snapshot(snapshot, false);
        if isfield(snapshot, 'truncated') && snapshot.truncated
            stop_timer_if_running(app.timer);
            set_status('Episode complete');
        end
    end

    function update_snapshot(snapshot, isReset)
        app = guidata(fig);
        apply_pose(app.twin, app.geom, snapshot.pose);
        title(app.axTwin, build_twin_title(snapshot), 'FontWeight', 'bold', 'FontName', 'Consolas');

        if ~isReset
            stepNum = double(snapshot.step_index);
            app.history.steps(end+1) = stepNum;
            if isfield(snapshot, 'reward')
                app.history.reward(end+1) = snapshot.reward;
            else
                app.history.reward(end+1) = NaN;
            end
            app.history.capture(end+1) = safe_nested(snapshot, {'reward_breakdown','capture_reward'}, NaN);
            app.history.temp(end+1) = safe_nested(snapshot, {'reward_breakdown','temp_penalty'}, NaN);
            app.history.correction(end+1) = safe_nested(snapshot, {'reward_breakdown','correction_penalty'}, NaN);
            app.history.driftT(end+1) = safe_nested(snapshot, {'discrepancy','T_operating'}, NaN);
            app.history.driftEta(end+1) = safe_nested(snapshot, {'discrepancy','eta'}, NaN);
            app.history.pitch(end+1) = safe_nested(snapshot, {'pose','pitch'}, NaN);
            app.history.yaw(end+1) = safe_nested(snapshot, {'pose','yaw'}, NaN);
            app.history.roll(end+1) = safe_nested(snapshot, {'pose','roll'}, NaN);
            app.history.z(end+1) = safe_nested(snapshot, {'pose','z'}, NaN);
        end

        cla(app.axReward); cla(app.axDrift); cla(app.axPose);
        plot(app.axReward, app.history.steps, app.history.reward, '-k', 'LineWidth', 1.2);
        plot(app.axReward, app.history.steps, app.history.capture, '--g');
        plot(app.axReward, app.history.steps, -app.history.temp, '--r');
        plot(app.axReward, app.history.steps, -app.history.correction, '--m');
        legend(app.axReward, {'total','capture','-temp','-correction'}, 'Location', 'best');
        grid(app.axReward, 'on');

        yyaxis(app.axDrift, 'left');
        p1 = plot(app.axDrift, app.history.steps, app.history.driftT, '-b', 'LineWidth', 1.2);
        ylabel(app.axDrift, '\DeltaT (K)');
        yyaxis(app.axDrift, 'right');
        p2 = plot(app.axDrift, app.history.steps, app.history.driftEta, '-r', 'LineWidth', 1.2);
        ylabel(app.axDrift, '\Delta\eta');
        legend(app.axDrift, [p1, p2], {'\DeltaT (K)','\Delta\eta'}, 'Location', 'best');
        grid(app.axDrift, 'on');

        plot(app.axPose, app.history.steps, app.history.pitch, '-b');
        plot(app.axPose, app.history.steps, app.history.yaw, '-r');
        plot(app.axPose, app.history.steps, app.history.roll, '-g');
        plot(app.axPose, app.history.steps, app.history.z, '-k');
        legend(app.axPose, {'pitch','yaw','roll','z'}, 'Location', 'best');
        grid(app.axPose, 'on');

        set(app.txtPose.body, 'String', format_pose(snapshot));
        set(app.txtPred.body, 'String', format_predictions(snapshot));
        set(app.txtReward.body, 'String', format_reward(snapshot));
        set(app.txtDecision.body, 'String', format_decision(snapshot));
        set(app.txtDrift.body, 'String', format_drift(snapshot));
        drawnow limitrate;
        guidata(fig, app);
    end

    function out = gather_conditions()
        app = guidata(fig);
        out = struct( ...
            'lat', str2double(app.inputs.lat.value.String), ...
            'lon', str2double(app.inputs.lon.value.String), ...
            'alt', str2double(app.inputs.alt.value.String), ...
            'day_of_year', round(str2double(app.inputs.day.value.String)), ...
            'month', round(str2double(app.inputs.month.value.String)), ...
            'year', round(str2double(app.inputs.year.value.String)), ...
            'hour', str2double(app.inputs.hour.value.String), ...
            'minute', str2double(app.inputs.minute.value.String), ...
            'ambient_c', str2double(app.inputs.ambient.value.String), ...
            'wind_mps', str2double(app.inputs.wind.value.String), ...
            'wind_dir', str2double(app.inputs.winddir.value.String), ...
            'humidity', str2double(app.inputs.humidity.value.String), ...
            'irradiance', str2double(app.inputs.irradiance.value.String), ...
            'cloud_cover', str2double(app.inputs.cloud.value.String), ...
            'pressure', str2double(app.inputs.pressure.value.String));
    end

    function out = gather_pose()
        app = guidata(fig);
        out = struct( ...
            'pitch', str2double(app.inputs.pitch.value.String), ...
            'yaw', str2double(app.inputs.yaw.value.String), ...
            'roll', str2double(app.inputs.roll.value.String), ...
            'z', str2double(app.inputs.z.value.String));
    end

    function set_status(msg)
        app = guidata(fig);
        set(app.txtStatus, 'String', msg);
        drawnow limitrate;
    end

    function tf = ensure_bridge_ready(showDialog)
        if nargin < 1
            showDialog = true;
        end
        app = guidata(fig);
        tf = isfield(app, 'pythonReady') && app.pythonReady && ~isempty(app.bridge);
        if tf
            return;
        end
        stop_timer_if_running(app.timer);
        if showDialog
            warndlg(['The Python runtime for run_simulation is not ready yet. ' ...
                     'Use Reset after fixing the setup issue, or restart MATLAB if the Python runtime changed.'], ...
                     'AURA-MPC simulator not ready', 'modal');
        end
    end

    function onClose(~, ~)
        app = guidata(fig);
        try
            stop_timer_if_running(app.timer);
            delete(app.timer);
        catch
        end
        delete(fig);
    end
end

function bridge = create_bridge(configPath, checkpointPath, outputDir, requirementsPath, venvDir)
    persistent cachedBridge cachedConfig cachedCheckpoint cachedOutput cachedPython
    here = fileparts(configPath);
    pkgRoot = fileparts(here);
    pythonExe = ensure_python_runtime(requirementsPath, venvDir);
    pyPaths = cell(py.sys.path);
    if ~any(strcmp(pyPaths, pkgRoot))
        insert(py.sys.path, int32(0), pkgRoot);
    end
    if isempty(cachedBridge) ...
            || ~strcmp(cachedConfig, configPath) ...
            || ~strcmp(cachedCheckpoint, checkpointPath) ...
            || ~strcmp(cachedOutput, outputDir) ...
            || ~strcmp(cachedPython, pythonExe)
        mod = py.importlib.import_module('sandbox.matlab_bridge');
        cachedBridge = mod.MatlabSimulationBridge(configPath, checkpointPath, py.None, outputDir, 'cpu');
        cachedConfig = configPath;
        cachedCheckpoint = checkpointPath;
        cachedOutput = outputDir;
        cachedPython = pythonExe;
    end
    bridge = cachedBridge;
end

function pythonExe = ensure_python_runtime(requirementsPath, venvDir)
    requiredModules = {'numpy', 'torch', 'yaml'};
    runtimePackages = { ...
        'numpy>=1.24,<2.0', ...
        'pyyaml>=6.0.1', ...
        'torch>=2.0.0'};
    pythonExe = venv_python_executable(venvDir);

    if ~isfile(pythonExe)
        bootstrap_local_venv(venvDir, runtimePackages);
    elseif ~python_has_modules(pythonExe, requiredModules)
        install_runtime_dependencies(pythonExe, runtimePackages, requirementsPath);
    end

    pe = pyenv;
    if strcmp(pe.Status, 'Loaded') && ~strcmp(char(pe.Version), pythonExe)
        try
            terminate(pe);
        catch err
            error(['run_simulation requires a dedicated Python environment at:\n  %s\n' ...
                   'MATLAB already has a different Python runtime loaded:\n  %s\n' ...
                   'Restart MATLAB, then run run_simulation again.\n\nOriginal error: %s'], ...
                   pythonExe, char(pe.Version), err.message);
        end
    end

    pe = pyenv;
    if ~strcmp(pe.Status, 'Loaded') || ~strcmp(char(pe.Version), pythonExe)
        pyenv('Version', pythonExe, 'ExecutionMode', 'OutOfProcess');
    end

    if ~python_has_modules(pythonExe, requiredModules)
        error(['Python environment is missing required modules even after setup.\n' ...
               'Expected interpreter:\n  %s\n' ...
               'Required modules: %s'], ...
               pythonExe, strjoin(requiredModules, ', '));
    end
end

function pythonExe = venv_python_executable(venvDir)
    if ispc
        pythonExe = fullfile(venvDir, 'Scripts', 'python.exe');
    else
        pythonExe = fullfile(venvDir, 'bin', 'python');
    end
end

function bootstrap_local_venv(venvDir, runtimePackages)
    if ~exist(fileparts(venvDir), 'dir')
        mkdir(fileparts(venvDir));
    end

    hostPython = find_host_python();
    [status, cmdout] = system(sprintf('"%s" -m venv "%s"', hostPython, venvDir));
    if status ~= 0
        error(['Failed to create the MATLAB Python virtual environment.\n' ...
               'Command output:\n%s'], cmdout);
    end

    pythonExe = venv_python_executable(venvDir);
    install_runtime_dependencies(pythonExe, runtimePackages, '');
end

function install_runtime_dependencies(pythonExe, runtimePackages, requirementsPath)
    [statusPip, outPip] = system(sprintf('"%s" -m pip install --upgrade pip', pythonExe));
    if statusPip ~= 0
        error(['Failed to upgrade pip for the MATLAB Python environment.\n' ...
               'Command output:\n%s'], outPip);
    end

    quotedPackages = cellfun(@(pkg) ['"' pkg '"'], runtimePackages, 'UniformOutput', false);
    installCmd = sprintf('"%s" -m pip install %s', pythonExe, strjoin(quotedPackages, ' '));
    [statusReq, outReq] = system(installCmd);
    if statusReq ~= 0
        if isempty(requirementsPath)
            requirementContext = 'Minimal runtime package set for run_simulation.';
        else
            requirementContext = sprintf(['Minimal runtime package set derived from run_simulation.\n' ...
                                          'Full PINN requirements were intentionally not used because they include ' ...
                                          'packages that may not be available for the MATLAB Python version.\n' ...
                                          'Reference requirements file:\n  %s'], requirementsPath);
        end
        error(['Failed to install Python dependencies for run_simulation.\n' ...
               '%s\n' ...
               'Command output:\n%s'], requirementContext, outReq);
    end
end

function tf = python_has_modules(pythonExe, moduleNames)
    quotedModules = cellfun(@(name) ['''' name ''''], moduleNames, 'UniformOutput', false);
    pythonSnippet = sprintf(['import importlib.util\n' ...
                             'mods=[%s]\n' ...
                             'missing=[m for m in mods if importlib.util.find_spec(m) is None]\n' ...
                             'raise SystemExit(0 if not missing else 1)\n'], ...
                             strjoin(quotedModules, ','));
    [status, ~] = system(sprintf('"%s" -c "%s"', pythonExe, escape_shell_double_quotes(pythonSnippet)));
    tf = (status == 0);
end

function pythonExe = find_host_python()
    if ispc
        candidates = {'py -3', 'python', 'python3'};
    else
        candidates = {'python3', 'python'};
    end

    for idx = 1:numel(candidates)
        probe = candidates{idx};
        [status, out] = system(sprintf('%s -c "import sys; print(sys.executable)"', probe));
        if status == 0
            pythonExe = strtrim(out);
            return;
        end
    end

    error(['Could not find a host Python interpreter to bootstrap run_simulation.\n' ...
           'Install Python 3, then rerun the simulator.']);
end

function out = escape_shell_double_quotes(text)
    out = strrep(text, '"', '\"');
end

function stop_timer_if_running(tmr)
    try
        if strcmp(tmr.Running, 'on')
            stop(tmr);
        end
    catch
    end
end

function h = empty_history()
    h = struct( ...
        'steps', [], ...
        'reward', [], ...
        'capture', [], ...
        'temp', [], ...
        'correction', [], ...
        'driftT', [], ...
        'driftEta', [], ...
        'pitch', [], ...
        'yaw', [], ...
        'roll', [], ...
        'z', []);
end

function ui = make_edit(parent, label, x, y, value)
    uicontrol(parent, 'Style', 'text', 'String', label, 'Position', [x y+20 140 18], ...
        'HorizontalAlignment', 'left', 'BackgroundColor', 'w');
    ui.value = uicontrol(parent, 'Style', 'edit', 'String', value, ...
        'Position', [x y 140 22], 'BackgroundColor', 'w');
end

function ui = make_output(parent, titleText, pos)
    ui.title = uicontrol(parent, 'Style', 'text', 'String', titleText, ...
        'Position', [pos(1) pos(2)+pos(4)-18 pos(3) 18], ...
        'HorizontalAlignment', 'left', 'BackgroundColor', 'w', 'FontWeight', 'bold');
    ui.body = uicontrol(parent, 'Style', 'edit', 'Max', 20, 'Min', 0, ...
        'Enable', 'inactive', 'HorizontalAlignment', 'left', ...
        'Position', [pos(1) pos(2) pos(3) pos(4)-22], ...
        'BackgroundColor', [0.98 0.98 0.98], 'String', '');
end

function text = build_twin_title(snapshot)
    pose = snapshot.pose;
    reward = NaN;
    if isfield(snapshot, 'reward')
        reward = snapshot.reward;
    end
    text = sprintf('step=%3d  reward=%+7.3f  pitch=%+5.1f  yaw=%+6.1f  roll=%+5.1f  z=%.2f m', ...
        double(snapshot.step_index), reward, pose.pitch, pose.yaw, pose.roll, pose.z);
end

function text = format_pose(snapshot)
    p = snapshot.pose;
    text = sprintf(['pitch: %7.2f deg\n' ...
                    'yaw:   %7.2f deg\n' ...
                    'roll:  %7.2f deg\n' ...
                    'z:     %7.3f m\n' ...
                    'time:  %02.0f:%05.2f day %d'], ...
        p.pitch, p.yaw, p.roll, p.z, ...
        snapshot.conditions.hour, snapshot.conditions.minute, snapshot.conditions.day_of_year);
end

function text = format_predictions(snapshot)
    pinn = snapshot.pinn_prediction;
    rk4 = [];
    if isfield(snapshot, 'rk4_prediction') && ~isempty(snapshot.rk4_prediction)
        rk4 = snapshot.rk4_prediction;
    end
    text = sprintf('PINN T: %.3f K\nPINN eta: %.5f\n', pinn.T_operating, pinn.eta);
    if ~isempty(rk4)
        text = sprintf('%sRK4 T:  %.3f K\nRK4 eta: %.5f\nG_eff:  %.3f W/m^2', ...
            text, rk4.T_operating, rk4.eta, rk4.G_eff);
    else
        corrected = snapshot.corrected_prediction;
        text = sprintf('%sCorrected T: %.3f K\nCorrected eta: %.5f', ...
            text, corrected.T_operating, corrected.eta);
    end
end

function text = format_reward(snapshot)
    if ~isfield(snapshot, 'reward_breakdown') || isempty(snapshot.reward_breakdown)
        text = 'Reward not computed yet.';
        return;
    end
    rb = snapshot.reward_breakdown;
    total = NaN;
    if isfield(snapshot, 'reward')
        total = snapshot.reward;
    else
        total = rb.total_reward;
    end
    text = sprintf(['origin: %s\n' ...
                    'capture:    %+0.4f\n' ...
                    'temp pen.:  %+0.4f\n' ...
                    'smooth pen: %+0.4f\n' ...
                    'corr. pen.: %+0.4f\n' ...
                    'total:      %+0.4f'], ...
        rb.reward_origin, rb.capture_reward, rb.temp_penalty, ...
        rb.smoothness_penalty, rb.correction_penalty, total);
end

function text = format_decision(snapshot)
    ctx = snapshot.policy_context;
    meanAction = safe_nested(snapshot, {'policy_context','action_mean'}, []);
    applied = safe_nested(snapshot, {'policy_context','action_applied'}, []);
    reason = snapshot.decision_reason;
    text = sprintf('mode: %s\nmean action: %s\napplied action: %s\npolicy updated: %d\n\n%s', ...
        ctx.mode, mat2str(meanAction, 3), mat2str(applied, 3), ...
        logical(safe_nested(snapshot, {'policy_updated'}, false)), reason);
end

function text = format_drift(snapshot)
    driftT = safe_nested(snapshot, {'discrepancy','T_operating'}, NaN);
    driftEta = safe_nested(snapshot, {'discrepancy','eta'}, NaN);
    biasT = safe_nested(snapshot, {'bias_correction','T_operating'}, NaN);
    biasEta = safe_nested(snapshot, {'bias_correction','eta'}, NaN);
    validated = safe_nested(snapshot, {'validation','performed'}, false);
    text = sprintf(['validated this step: %d\n' ...
                    'drift dT: %.3f K   drift dEta: %.5f\n' ...
                    'EMA bias T: %.3f K   EMA bias eta: %.5f'], ...
        logical(validated), driftT, driftEta, biasT, biasEta);
end

function value = safe_nested(s, keys, defaultValue)
    value = defaultValue;
    current = s;
    for i = 1:numel(keys)
        key = keys{i};
        if isstruct(current) && isfield(current, key)
            current = current.(key);
        else
            return;
        end
    end
    value = current;
end

function [geom, twin] = build_twin(ax, here)
    geom = struct( ...
        'BASE_THICK', 10, ...
        'LOWER_H', 220, ...
        'UPPER_H', 160, ...
        'UPPER_TRAVEL', 130, ...
        'YAW_THICK', 16, ...
        'ARM_H', 55, ...
        'ARM_BASE_THICK', 8, ...
        'PIVOT_BOSS_D', 12, ...
        'PM_THICK', 6, ...
        'PM_LUG_H', 14, ...
        'PANEL_Z', 70, ...
        'YAW_SEAT_Z', 88);

    meshes.base = stlread(fullfile(here, 'AURA_MFP_base_plate.stl'));
    meshes.post_outer = stlread(fullfile(here, 'AURA_MFP_post_outer.stl'));
    meshes.post_inner = stlread(fullfile(here, 'AURA_MFP_post_inner.stl'));
    meshes.yaw_disk = stlread(fullfile(here, 'AURA_MFP_yaw_disk.stl'));
    meshes.pitch_b = stlread(fullfile(here, 'AURA_MFP_pitch_bracket.stl'));
    meshes.roll_b = stlread(fullfile(here, 'AURA_MFP_roll_bracket.stl'));
    meshes.panel_plate = stlread(fullfile(here, 'AURA_MFP_panel_mount_plate.stl'));

    twin.t_base = hgtransform('Parent', ax);
    twin.t_post_outer = hgtransform('Parent', twin.t_base);
    twin.t_post_inner = hgtransform('Parent', twin.t_post_outer);
    twin.t_yaw = hgtransform('Parent', twin.t_post_inner);
    twin.t_pitch_base = hgtransform('Parent', twin.t_yaw);
    twin.t_pitch_arm = hgtransform('Parent', twin.t_yaw);
    twin.t_roll_base = hgtransform('Parent', twin.t_pitch_arm);
    twin.t_roll_arm = hgtransform('Parent', twin.t_pitch_arm);
    twin.t_panel = hgtransform('Parent', twin.t_roll_arm);

    render_part(meshes.base, twin.t_base, [0.28 0.46 0.63]);
    render_part(meshes.post_outer, twin.t_post_outer, [0.69 0.77 0.87]);
    render_part(meshes.post_inner, twin.t_post_inner, [0.39 0.58 0.93]);
    render_part(meshes.yaw_disk, twin.t_yaw, [0.12 0.56 1.00]);
    render_part(meshes.pitch_b, twin.t_pitch_base, [0.25 0.41 0.88]);
    render_part(meshes.roll_b, twin.t_roll_base, [0.00 0.00 0.50]);
    render_part(meshes.panel_plate, twin.t_panel, [0.10 0.10 0.44]);

    set(twin.t_post_outer, 'Matrix', makehgtform('translate', [0 0 geom.BASE_THICK + geom.LOWER_H/2]));
    set(twin.t_pitch_base, 'Matrix', makehgtform('translate', [0 0 geom.YAW_THICK/2]));
    set(twin.t_roll_base, 'Matrix', makehgtform('xrotate', pi));
    set(twin.t_panel, 'Matrix', ...
        makehgtform('translate', [0 0 geom.PM_THICK + geom.PM_LUG_H]) * ...
        makehgtform('xrotate', pi));
end

function apply_pose(twin, geom, pose)
    zMin = 0.5;
    zMax = 2.0;
    h = max(0.0, min(geom.UPPER_TRAVEL, geom.UPPER_TRAVEL * (pose.z - zMin) / (zMax - zMin)));
    set(twin.t_post_inner, 'Matrix', makehgtform('translate', [0 0 43.8 + h]));
    set(twin.t_yaw, 'Matrix', ...
        makehgtform('translate', [0 0 geom.YAW_SEAT_Z]) * ...
        makehgtform('zrotate', deg2rad(pose.yaw)));
    set(twin.t_pitch_arm, 'Matrix', ...
        makehgtform('translate', [0 0 geom.ARM_BASE_THICK + geom.ARM_H - geom.PIVOT_BOSS_D/2]) * ...
        makehgtform('xrotate', deg2rad(pose.pitch)));
    set(twin.t_roll_arm, 'Matrix', ...
        makehgtform('translate', [0 0 geom.ARM_H - geom.PIVOT_BOSS_D/2]) * ...
        makehgtform('yrotate', deg2rad(pose.roll)));
end

function render_part(mesh_data, parent_handle, color_vec)
    patch('Faces', mesh_data.ConnectivityList, ...
          'Vertices', mesh_data.Points, ...
          'Parent', parent_handle, ...
          'FaceColor', color_vec, ...
          'EdgeColor', 'none', ...
          'FaceLighting', 'gouraud', ...
          'AmbientStrength', 0.4, ...
          'DiffuseStrength', 0.6, ...
          'SpecularStrength', 0.2);
end
