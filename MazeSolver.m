function varargout = MazeSolver(varargin)
% MAZESOLVER MATLAB code for MazeSolver.fig
%       MAZESOLVER, by itself, creates a new MAZESOLVER or raises the existing
%       singleton*.
%
%       H = MAZESOLVER returns the handle to a new MAZESOLVER or the handle to
%       the existing singleton*.
%
%       MAZESOLVER('CALLBACK',hObject,eventData,handles,...) calls the local
%       function named CALLBACK in MAZESOLVER.M with the given input arguments.
%
%       MAZESOLVER('Property','Value',...) creates a new MAZESOLVER or raises the
%       existing singleton*.  Starting from the left, property value pairs are
%       applied to the GUI before MazeSolver_OpeningFcn gets called.  An
%       unrecognized property name or invalid value makes property application
%       stop. All inputs are passed to MazeSolver_OpeningFcn via varargin.
%
%       *See GUI Options on GUIDE's Tools menu. Choose "GUI allows only one
%       instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES
% Edit the above text to modify the response to help MazeSolver
% Last Modified by GUIDE v2.5 22-May-2025 20:36:18
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MazeSolver_OpeningFcn, ...
                   'gui_OutputFcn',  @MazeSolver_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end
if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before MazeSolver is made visible.
function MazeSolver_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MazeSolver (see VARARGIN)

% Choose default command line output for MazeSolver
handles.output = hObject;

% 初始化变量
handles.mazeImage = [];
handles.binaryMaze = [];
handles.startPoint = [];
handles.endPoint = [];
handles.path = [];
handles.waitingForStart = false;
handles.waitingForEnd = false;
handles.distanceMatrix = []; % 用于Dijkstra算法的邻接矩阵/距离矩阵
handles.pixelGraph = [];     % 迷宫的图对象
handles.pathWidth = 1;
handles.useGPU = false;      % 此变量似乎未被直接使用，GPU选择由复选框控制
handles.gpuAvailable = false;

% 添加存储原始迷宫尺寸的字段，用于坐标检查
handles.originalMazeHeight = 0;
handles.originalMazeWidth = 0;

% 检查 GPU 可用性
try
    if gpuDeviceCount > 0
        handles.gpuAvailable = true;
        gpu = gpuDevice();
        gpuInfo = sprintf('GPU可用: %s (%.1f GB)', gpu.Name, gpu.AvailableMemory/1e9);
        set(handles.gpuCheckbox, 'Enable', 'on'); % 启用 GPU 复选框
        set(handles.gpuInfoText, 'String', gpuInfo); % 更新 GPU 信息文本
    else
        gpuInfo = 'GPU不可用';
        set(handles.gpuInfoText, 'String', gpuInfo);
        set(handles.gpuCheckbox, 'Enable', 'off'); % 禁用 GPU 复选框
        set(handles.gpuCheckbox, 'Value', 0); % 确保未选中
    end
catch
    gpuInfo = 'GPU不可用';
    set(handles.gpuInfoText, 'String', gpuInfo);
    set(handles.gpuCheckbox, 'Enable', 'off'); % 禁用 GPU 复选框
    set(handles.gpuCheckbox, 'Value', 0); % 确保未选中
end

% 设置初始状态
set(handles.statusText, 'String', '就绪 - 请先加载迷宫图像');
set(handles.figure1, 'WindowButtonDownFcn', {@figure_ButtonDownFcn, handles});

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MazeSolver wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MazeSolver_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on mouse press over figure window.
function figure_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = guidata(hObject); % 确保获取最新的 handles

if isempty(handles.mazeImage)
    set(handles.statusText, 'String', '错误：请先加载迷宫图像！');
    return;
end

if ~handles.waitingForStart && ~handles.waitingForEnd
    return; % 非等待状态，不处理点击
end

% Get the current clicked point (relative to mazeAxes)
clickPoint = get(handles.mazeAxes, 'CurrentPoint');
x = round(clickPoint(1, 1));
y = round(clickPoint(1, 2));

% Ensure the point is within image boundaries
[height, width] = size(handles.binaryMaze);
if x < 1 || x > width || y < 1 || y > height
    set(handles.statusText, 'String', '错误：点击位置超出迷宫范围！');
    return;
end

% 检查在 binaryMaze 中，该点是否为墙
% handles.binaryMaze 的尺寸与 originalMazeHeight/Width 相同，且其最外层已被强制设置为墙
if handles.binaryMaze(y, x) == 0 % 墙体是0，路径是1
    set(handles.statusText, 'String', '错误：不能在墙上选点！');
    return;
end

if handles.waitingForStart
    handles.startPoint = [y, x]; % 存储原始坐标
    set(handles.startPointLabel, 'String', sprintf('起点: (%d, %d)', x, y));
    handles.waitingForStart = false;
    updateMazeDisplay(handles); % 更新显示，标记起点
    set(handles.statusText, 'String', '成功设置起点，请设置终点或求解。');
elseif handles.waitingForEnd
    handles.endPoint = [y, x]; % 存储原始坐标
    set(handles.endPointLabel, 'String', sprintf('终点: (%d, %d)', x, y));
    handles.waitingForEnd = false;
    updateMazeDisplay(handles); % 更新显示，标记终点
    set(handles.statusText, 'String', '成功设置终点，可以开始求解。');
end

guidata(hObject, handles); % 保存更新后的 handles

% --- 增强型迷宫求解主回调函数
function solveMazeButton_Callback(hObject, eventdata, handles)
% hObject    handle to solveMazeButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = guidata(hObject);

if isempty(handles.mazeImage)
    set(handles.statusText, 'String', '错误：请先加载迷宫图像！');
    return;
end

if isempty(handles.startPoint) || isempty(handles.endPoint)
    set(handles.statusText, 'String', '错误：请先设置起点和终点！');
    return;
end

try
    algorithmIndex = get(handles.algorithmPopup, 'Value');
    algorithmNames = {'Dijkstra (经典)', 'A* (启发式)', 'BFS (广度优先)'};
    selectedAlgorithmName = algorithmNames{algorithmIndex}; % 用于性能显示

    useGPU = get(handles.gpuCheckbox, 'Value') && handles.gpuAvailable;

    % binaryMaze 已经包含了强制的墙体边框，所以 height 和 width 应该是 binaryMaze 的尺寸
    [height, width] = size(handles.binaryMaze);
    
    % GUI选择的起点/终点是原始图像坐标，不需要额外 +1 映射到 binaryMaze
    % 因为 buildMazeGraph_with_border 已经将原始坐标映射到图中，
    % 而 binaryMaze 的尺寸就是原始图像的尺寸，只是其边缘值被强制置0
    startPos_current = handles.startPoint; % (y, x)
    endPos_current = handles.endPoint;   % (y, x)

    startNode = sub2ind([height, width], startPos_current(1), startPos_current(2)); % Dijkstra/BFS/A*通用
    endNode = sub2ind([height, width], endPos_current(1), endPos_current(2));   % Dijkstra/BFS/A*通用
    
    % 再次检查起点终点是否在墙上，使用 current 坐标 (因为 binaryMaze 已经对应原始尺寸并强制边框)
    if handles.binaryMaze(startPos_current(1), startPos_current(2)) == 0 || handles.binaryMaze(endPos_current(1), endPos_current(2)) == 0
        set(handles.statusText, 'String', '错误：起点或终点不能在墙上！');
        return;
    end

    statusMsg = sprintf('使用%s算法计算中...', selectedAlgorithmName);
    % 仅当选择的算法可以受益于GPU时才添加GPU提示
    if useGPU && (algorithmIndex == 1 || algorithmIndex == 2 || algorithmIndex == 3) 
        statusMsg = [statusMsg, ' (尝试GPU加速)'];
    end
    set(handles.statusText, 'String', statusMsg);
    drawnow; % 强制更新GUI

    tic; % 开始计时
    pathIndices = []; % 初始化路径索引

    switch algorithmIndex
        case 1 % Dijkstra
            if isempty(handles.distanceMatrix)
                set(handles.statusText, 'String', '错误: Dijkstra 算法需要距离矩阵，请重新加载迷宫或检查buildMazeGraph_with_border函数。');
                return;
            end
            if useGPU
                fprintf('信息: 尝试Dijkstra算法，距离矩阵使用gpuArray输入。\n');
                % 确保 distanceMatrix 已经转换为双精度，以便在 GPU 上使用
                distanceMatrix_gpu = gpuArray(double(handles.distanceMatrix)); 
                [~, pathIndices_result] = dijkstra_enhanced(distanceMatrix_gpu, startNode, endNode);
                pathIndices = gather(pathIndices_result); % 确保结果在CPU上
            else
                [~, pathIndices] = dijkstra_enhanced(handles.distanceMatrix, startNode, endNode);
            end

        case 2 % A*
            heuristicWeight = str2double(get(handles.heuristicWeight, 'String'));
            if isnan(heuristicWeight) || heuristicWeight <= 0
                heuristicWeight = 1.0; % 默认或有效权重
                set(handles.heuristicWeight, 'String', '1.0'); % 更新GUI显示
            end
            if useGPU
                fprintf('信息: 尝试A*算法，迷宫数据使用gpuArray输入。\n');
                % binaryMaze 已经是 logical 类型，直接传递即可
                binaryMaze_gpu = gpuArray(handles.binaryMaze); 
                pathIndices_result = astar_enhanced(binaryMaze_gpu, startPos_current, endPos_current, heuristicWeight);
                pathIndices = gather(pathIndices_result);
            else
                pathIndices = astar_enhanced(handles.binaryMaze, startPos_current, endPos_current, heuristicWeight);
            end

        case 3 % BFS
            if useGPU
                fprintf('信息: 尝试BFS算法，迷宫数据使用gpuArray输入。\n');
                binaryMaze_gpu = gpuArray(handles.binaryMaze);
                pathIndices_result = bfs_enhanced(binaryMaze_gpu, startPos_current, endPos_current);
                pathIndices = gather(pathIndices_result);
            else
                pathIndices = bfs_enhanced(handles.binaryMaze, startPos_current, endPos_current);
            end
    end

    solveTime = toc; % 结束计时

    % 将路径索引转换为原始坐标并确定路径长度
    if isempty(pathIndices)
        path = [];
        pathLength = 0;
    else
        % 将 pathIndices 转换为 (y, x) 坐标
        pathCoords = convertPathToCoords(pathIndices, height, width);
        path = pathCoords; 
        pathLength = size(path, 1); % 路径中的步数/节点数
    end

    if isempty(path)
        set(handles.statusText, 'String', '没有找到可行路径！');
        handles.path = []; % 清空路径
    else
        % 如果启用了路径平滑
        if get(handles.smoothingCheckbox, 'Value')
            % smoothPath 内部会处理坐标和 binaryMaze 的关系
            path = smoothPath(path, handles.binaryMaze); 
        end

        handles.path = path;
        updateMazeDisplay(handles); % 更新迷宫显示（包括路径）

        % 更新性能信息
        memInfo = memory; % 获取内存使用情况
        perfText = sprintf('算法: %s,时间: %.3f秒,内存: %.1fMB,路径长度: %d', ...
            selectedAlgorithmName, solveTime, memInfo.MemUsedMATLAB/1e6, pathLength);
        set(handles.performanceText, 'String', perfText);

        set(handles.statusText, 'String', sprintf('已找到路径，用时%.3f秒', solveTime));
    end

catch ME
    set(handles.statusText, 'String', sprintf('错误：%s (发生在 %s 第 %d 行)', ME.message, ME.stack(1).name, ME.stack(1).line));
    fprintf(2, '错误详情：\n'); % 打印错误到命令行，方便调试
    disp(ME);
end
guidata(hObject, handles); % 保存更新后的 handles


% --- 核心算法实现 ---

function [dist, path] = dijkstra_enhanced(distMatrix, startNode, endNode)
% 优化的 Dijkstra 算法 (当前使用线性搜索模拟优先队列)
% distMatrix: 邻接矩阵，其中 distMatrix(i,j) 是从节点i到节点j的权重
% startNode: 起始节点索引
% endNode: 目标节点索引

% 尝试确定节点数量
if isa(distMatrix, 'gpuArray')
    % 对于 gpuArray，size 可能会返回 gpuArray，需要 gather 来获取 CPU 上的值
    s = size(gather(distMatrix)); 
else
    s = size(distMatrix);
end
n = s(1);

if isa(distMatrix, 'gpuArray')
    dist = gpuArray.inf(n, 1);
    prev = gpuArray.zeros(n, 1, 'uint32'); % prev 存储索引，使用 uint32 更高效
    visited = gpuArray.false(n, 1);
else
    dist = inf(n, 1);
    prev = zeros(n, 1, 'uint32');
    visited = false(n, 1);
end

dist(startNode) = 0;

while true
    unvisitedDist = dist;
    unvisitedDist(visited) = inf; % 忽略已访问的节点
    
    % 找到未访问节点中的最小距离
    if isa(unvisitedDist, 'gpuArray')
        % gpuArray 的 min 函数对 inf 值处理正常
        [minDist_val, u_temp] = min(unvisitedDist); 
        u = u_temp(1); % 取第一个索引，如果多个相同最小值
    else
        [minDist_val, u] = min(unvisitedDist);
    end
    
    if isinf(minDist_val)
        break; % 所有可达节点都已访问
    end
    
    if u == endNode
        break;
    end
    
    visited(u) = true;
    
    % 找到当前节点 u 的邻居
    % sparse 矩阵的 find 方法可以处理 gpuArray，返回的索引也会是 gpuArray
    if isa(distMatrix, 'gpuArray')
        [~, neighbors_gpu, weights_gpu] = find(distMatrix(u, :));
        neighbors = gather(neighbors_gpu);
        weights = gather(weights_gpu);
    else
        [~, neighbors, weights] = find(distMatrix(u, :));
    end
    
    % 遍历邻居节点
    for i = 1:length(neighbors)
        v = neighbors(i);
        edgeWeight = weights(i); % 获取边的权重

        % 确保 v 是有效的且未访问的节点
        if v >= 1 && v <= n && ~visited(v)
            alt = dist(u) + edgeWeight;
            if alt < dist(v)
                dist(v) = alt;
                prev(v) = u;
            end
        end
    end
end

% 重建路径 (确保在 CPU 上进行)
path = [];
% 确保 dist 和 prev 是在 CPU 上进行路径重建，如果它们是 gpuArray
dist_cpu = gather(dist);
prev_cpu = gather(prev);
startNode_cpu = gather(startNode);
endNode_cpu = gather(endNode);

if ~isinf(dist_cpu(endNode_cpu))
    path = endNode_cpu;
    current = endNode_cpu;
    
    loop_count = 0;
    max_loop = n; % 最长路径不应超过节点总数
    
    while current ~= startNode_cpu && prev_cpu(current) ~= 0 && loop_count < max_loop
        current = prev_cpu(current);
        path = [current, path];
        loop_count = loop_count + 1;
    end
    
    % 如果循环因达到最大迭代次数而终止，且未到达起点
    if loop_count == max_loop && current ~= startNode_cpu
        warning('Dijkstra路径重建可能陷入死循环或路径过长。');
        path = []; % 路径重建失败
    elseif ~isempty(path) && path(1) ~= startNode_cpu % 确保路径从起点开始
        if current == startNode_cpu
            path = [startNode_cpu, path];
        else % 没能回到起点，说明路径不完整或有问题
            path = [];
        end
    end
end
% 如果路径只包含终点但起点终点不一致，说明没找到有效路径
if length(path) == 1 && startNode_cpu ~= endNode_cpu
    path = [];
end


function [path] = astar_enhanced(binaryMaze, startPos, endPos, heuristicWeight)
% A* 算法，使用曼哈顿距离作为启发函数
% binaryMaze: 带强制墙体边框的二值迷宫 (logical 或 gpuArray logical)
% startPos, endPos: 起点/终点坐标 [y, x] (基于迷宫的原始尺寸)

[height, width] = size(binaryMaze);
startNode = sub2ind([height, width], startPos(1), startPos(2));
endNode = sub2ind([height, width], endPos(1), endPos(2));
numNodes = height * width;

% 使用 gpuArray 或常规数组
if isa(binaryMaze, 'gpuArray')
    openSet_fScores = gpuArray.inf(numNodes, 1);
    closedSet = gpuArray.false(numNodes, 1);
    gScore = gpuArray.inf(numNodes, 1);
    cameFrom = gpuArray.zeros(numNodes, 1, 'uint32');
else
    openSet_fScores = inf(numNodes, 1);
    closedSet = false(numNodes, 1);
    gScore = inf(numNodes, 1);
    cameFrom = zeros(numNodes, 1, 'uint32');
end

gScore(startNode) = 0;
fScore_start = heuristicWeight * manhattanDistance(startPos, endPos);
openSet_fScores(startNode) = fScore_start;

directions = [-1 0; 0 1; 1 0; 0 -1]; % 上右下左

numProcessed = 0; % 用于防止潜在的无限循环或异常情况
maxIterations = numNodes * 2; % 设定一个合理的上限

while true
    numProcessed = numProcessed + 1;
    if numProcessed > maxIterations
        warning('A* 算法处理节点过多，可能存在问题。');
        path = []; 
        return;
    end

    % 找到 openSet 中 fScore 最小的节点
    [min_fScore, current_idx_temp] = min(openSet_fScores);
    
    % 如果所有节点都已处理或不可达
    if isinf(min_fScore) || isempty(current_idx_temp)
        break;
    end
    
    current = current_idx_temp(1); % 取第一个索引，如果多个相同最小值

    % 如果找到终点
    if current == endNode
        break;
    end
    
    openSet_fScores(current) = inf; % 从 openSet 中移除
    closedSet(current) = true;     % 加入 closedSet

    [currentY, currentX] = ind2sub([height, width], current);

    % 遍历邻居
    for d = 1:4
        newY = currentY + directions(d, 1);
        newX = currentX + directions(d, 2);
        
        % 检查邻居是否在迷宫范围内
        if newY >= 1 && newY <= height && newX >= 1 && newX <= width
            % binaryMaze 可能是 gpuArray，需要 gather 来访问单个元素
            isWall = (gather(binaryMaze(newY, newX)) == 0); 
            
            if ~isWall
                neighbor = sub2ind([height, width], newY, newX);
                
                % 如果邻居已经在 closedSet 中，跳过
                if closedSet(neighbor)
                    continue;
                end
                
                tentativeGScore = gScore(current) + 1; % 步长为1
                
                % 如果找到更优的路径到达邻居
                if tentativeGScore < gScore(neighbor)
                    cameFrom(neighbor) = current;
                    gScore(neighbor) = tentativeGScore;
                    % 注意：这里 manhattanDistance 的输入是 CPU 上的数据，
                    % 如果 newY/newX 是 gpuArray，则需要 gather
                    hScore = heuristicWeight * manhattanDistance([newY, newX], endPos);
                    new_fScore = tentativeGScore + hScore;

                    % 如果邻居不在 openSet 中，或者新的 fScore 更小，则更新
                    openSet_fScores(neighbor) = new_fScore;
                end
            end
        end
    end
end

% 重建路径 (确保在 CPU 上进行)
path = [];
% 确保 cameFrom 在 CPU 上
cameFrom_cpu = gather(cameFrom); 
startNode_cpu = gather(startNode);
endNode_cpu = gather(endNode);

if closedSet(endNode) || (~isinf(gScore(endNode)) && gScore(endNode) > 0) % 终点已访问或可达
    path = endNode_cpu;
    current = endNode_cpu;
    loop_count = 0;
    max_loop = numNodes;

    while current ~= startNode_cpu && cameFrom_cpu(current) ~= 0 && loop_count < max_loop
        current = cameFrom_cpu(current);
        path = [current, path];
        loop_count = loop_count + 1;
    end
    
    if loop_count == max_loop && current ~= startNode_cpu
        warning('A*路径重建可能陷入死循环或路径过长。');
        path = [];
    elseif ~isempty(path) && path(1) ~= startNode_cpu % 确保路径从起点开始
        if current == startNode_cpu
            path = [startNode_cpu, path];
        else % 没能回到起点，说明路径不完整或有问题
            path = [];
        end
    end
end
% 如果路径只包含终点但起点终点不一致，说明没找到有效路径
if length(path) == 1 && startNode_cpu ~= endNode_cpu
    path = [];
end


function [path] = bfs_enhanced(binaryMaze, startPos, endPos)
% 广度优先搜索 (BFS)
% binaryMaze: 带强制墙体边框的二值迷宫 (logical 或 gpuArray logical)
% startPos, endPos: 起点/终点坐标 [y, x] (基于迷宫的原始尺寸)

[height, width] = size(binaryMaze);
startNode = sub2ind([height, width], startPos(1), startPos(2));
endNode = sub2ind([height, width], endPos(1), endPos(2));
numNodes = height * width;

if isa(binaryMaze, 'gpuArray')
    queue = gpuArray.zeros(1, numNodes, 'uint32'); % 队列存储索引
    visited = gpuArray.false(numNodes, 1);
    parent = gpuArray.zeros(numNodes, 1, 'uint32'); % parent 存储索引
else
    queue = zeros(1, numNodes, 'uint32');
    visited = false(numNodes, 1);
    parent = zeros(numNodes, 1, 'uint32');
end

qFront = 1;
qRear = 0;

qRear = qRear + 1;
queue(qRear) = startNode;
visited(startNode) = true;

directions = [-1 0; 0 1; 1 0; 0 -1]; % 上右下左

pathFound = false;

while qFront <= qRear
    current = queue(qFront);
    qFront = qFront + 1;
    
    if current == endNode
        pathFound = true;
        break;
    end
    
    [currentY, currentX] = ind2sub([height, width], current);
    
    for d = 1:4
        newY = currentY + directions(d, 1);
        newX = currentX + directions(d, 2);
        
        if newY >= 1 && newY <= height && newX >= 1 && newX <= width
            isWall = (gather(binaryMaze(newY, newX)) == 0);
            if ~isWall
                neighbor = sub2ind([height, width], newY, newX);
                if ~visited(neighbor)
                    visited(neighbor) = true;
                    parent(neighbor) = current;
                    qRear = qRear + 1;
                    queue(qRear) = neighbor;
                end
            end
        end
    end
end

% 重建路径 (确保在 CPU 上进行)
path = [];
if pathFound
    % 确保 parent 在 CPU 上
    parent_cpu = gather(parent); 
    startNode_cpu = gather(startNode);
    endNode_cpu = gather(endNode);

    path = endNode_cpu;
    current_path_node = endNode_cpu;
    loop_count = 0; 
    max_loop = numNodes; % 防止死循环

    while parent_cpu(current_path_node) ~= 0 && loop_count < max_loop
        current_path_node = parent_cpu(current_path_node);
        path = [current_path_node, path];
        loop_count = loop_count + 1;
    end
    
    if loop_count == max_loop && parent_cpu(current_path_node) ~= 0
         warning('BFS路径重建可能陷入死循环或路径过长。'); 
         path = [];
    elseif ~isempty(path) && path(1) ~= startNode_cpu % 确保路径从起点开始
        if current_path_node == startNode_cpu
            path = [startNode_cpu, path];
        else % 没能回到起点，说明路径不完整或有问题
            path = [];
        end
    end
end
% 如果路径只包含终点但起点终点不一致，说明没找到有效路径
if length(path) == 1 && startNode_cpu ~= endNode_cpu
    path = [];
end

% --- 辅助函数 ---

function dist = manhattanDistance(pos1, pos2)
% 计算曼哈顿距离
% pos1, pos2 格式为 [y, x]
dist = abs(pos1(1) - pos2(1)) + abs(pos1(2) - pos2(2));


function pathCoords = convertPathToCoords(pathIndices, mapHeight, mapWidth)
% 将线性索引转换为 (y, x) 坐标
if isempty(pathIndices)
    pathCoords = [];
    return;
end

% 确保 pathIndices 在 CPU 上进行转换
if isa(pathIndices, 'gpuArray')
    pathIndices_cpu = gather(pathIndices);
else
    pathIndices_cpu = pathIndices;
end

pathCoords = zeros(length(pathIndices_cpu), 2);
for i = 1:length(pathIndices_cpu)
    [row, col] = ind2sub([mapHeight, mapWidth], pathIndices_cpu(i));
    pathCoords(i, :) = [row, col];
end


function smoothedPath = smoothPath(path, binaryMaze)
% 路径平滑函数，通过跳过中间可直达的点来减少路径点数
% path: 原始路径，格式为 [y, x] 坐标 (基于原始图像坐标)
% binaryMaze: 带强制墙体边框的二值迷宫 (logical)
if size(path, 1) < 3
    smoothedPath = path;
    return;
end

smoothedPath = [path(1, :)]; % 起点总是包含

i = 1;
while i < size(path, 1)
    j = i + 2; % 尝试跳过至少一个点
    lastClearPointIndex = i + 1; % 初始时，下一个点是可达的
    
    while j <= size(path, 1)
        % isDirectPathClear 期望的是原始坐标，它会自行处理与 binaryMaze 的关系
        startCoord = path(i, :);
        goalCoord = path(j, :);

        % 这里不需要 +1，因为 isDirectPathClear 会使用 binaryMaze 
        % binaryMaze 已经对应了原始坐标，且其边缘是墙
        if isDirectPathClear(startCoord, goalCoord, binaryMaze)
            lastClearPointIndex = j; % 记录可直达的最远点
            j = j + 1;
        else
            break; % 无法直达，停止向前探索
        end
    end
    
    smoothedPath = [smoothedPath; path(lastClearPointIndex, :)]; % 添加可直达的最远点
    i = lastClearPointIndex; % 从这个点继续向前
    
    if i == size(path,1) % 如果已经到达终点
        break;
    end
end

% 确保终点被包含
if ~isequal(smoothedPath(end,:), path(end,:)) 
    % isDirectPathClear 期望的是原始坐标
    if isDirectPathClear(smoothedPath(end,:), path(end,:), binaryMaze) 
        smoothedPath = [smoothedPath; path(end,:)];
    end
end


function isClear = isDirectPathClear(startCoord, goalCoord, binaryMaze)
% 检查两点之间是否存在无障碍的直线路径 (使用 Bresenham's line algorithm)
% startCoord, goalCoord: 坐标 [y, x] (基于迷宫的原始尺寸)
% binaryMaze: 带强制墙体边框的二值迷宫 (logical)

isClear = true;
x1 = startCoord(2); y1 = startCoord(1);
x2 = goalCoord(2); y2 = goalCoord(1);

dx = abs(x2 - x1);
dy = abs(y2 - y1);
sx = sign(x2 - x1); % x 方向步进
sy = sign(y2 - y1); % y 方向步进
err = dx - dy;

x = x1; 
y = y1;

[mazeHeight, mazeWidth] = size(binaryMaze);

while true
    % 检查当前像素是否在迷宫范围内
    if y < 1 || y > mazeHeight || x < 1 || x > mazeWidth
        isClear = false;
        break;
    end
    
    % 检查当前像素是否是墙 (0 表示墙)
    % binaryMaze 可能是 gpuArray，需要 gather
    if gather(binaryMaze(y, x)) == 0
        isClear = false;
        break;
    end
    
    if x == x2 && y == y2 % 到达目标点
        break;
    end
    
    e2 = 2 * err;
    if e2 > -dy
        err = err - dy;
        x = x + sx;
    end
    if e2 < dx
        err = err + dx;
        y = y + sy;
    end
end


% Helper function to update the maze display with start point, end point, and path
function updateMazeDisplay(handles)
% Create a colored display image
if size(handles.mazeImage, 3) == 1
    % Convert to RGB if it's grayscale
    dispImage = repmat(handles.mazeImage, [1 1 3]);
else
    dispImage = handles.mazeImage;
end

if isa(dispImage, 'uint8')
    imgToShow = double(dispImage) / 255; % Normalize for imshow if it's uint8
else
    imgToShow = dispImage; % already normalized
end

% Display the updated image
axes(handles.mazeAxes);
imshow(imgToShow);
hold on; % Hold on to draw on top of the image

% Draw start point if it exists
if ~isempty(handles.startPoint)
    startY = handles.startPoint(1);
    startX = handles.startPoint(2);
    % Use plot function to draw a larger circle for the start point
    plot(startX, startY, 'o', 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'green', 'MarkerSize', 10);
end

% Draw end point if it exists
if ~isempty(handles.endPoint)
    endY = handles.endPoint(1);
    endX = handles.endPoint(2);
    % Use plot function to draw a larger circle for the end point
    plot(endX, endY, 'o', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'red', 'MarkerSize', 10);
end

% Draw the path if it exists
if ~isempty(handles.path)
    % Convert path coordinates to X and Y arrays (columns are X, rows are Y)
    pathX = handles.path(:, 2); % Column is X
    pathY = handles.path(:, 1); % Row is Y
    % Use plot function to draw the line segment
    plot(pathX, pathY, 'b-', 'LineWidth', handles.pathWidth); % Draw with a fixed width, slightly adjusted for visibility
end

hold off; % Release hold
title('迷宫求解');


% Helper function to build the maze graph and distance matrix
% This function now takes an original binary maze and forces its outer border to be walls.
function [pixelGraph, distMatrix] = buildMazeGraph_with_border(binaryMaze_input)
% Use a temporary copy for modification
binaryMaze = binaryMaze_input;

% Get maze dimensions
[height, width] = size(binaryMaze);

% **强制最外层边框为墙 (0)**
% 这是防止路径走到迷宫“外部”的关键步骤。
if height > 0 && width > 0
    binaryMaze(1, :) = 0; % 设置第一行为墙
    binaryMaze(height, :) = 0; % 设置最后一行为墙
    binaryMaze(:, 1) = 0; % 设置第一列为墙
    binaryMaze(:, width) = 0; % 设置最后一列为墙
end

numNodes = height * width;

% 初始化稀疏矩阵以提高效率
nnz_estimate = numNodes * 4; % 估算非零元素的数量 (每个像素最多有4个邻居)
% 为了防止预分配数组过小导致错误，可以预分配更大的数组或在需要时增加
row_indices = zeros(nnz_estimate, 1);
col_indices = zeros(nnz_estimate, 1);
values = zeros(nnz_estimate, 1);
count = 0; % 稀疏矩阵元素的计数器

% 定义4连接邻域 (上, 右, 下, 左)
directions = [-1 0; 0 1; 1 0; 0 -1]; % [dy dx]

% 构建邻接矩阵
for y = 1:height
    for x = 1:width
        % 如果当前位置是墙 (包括强制的边框)，则跳过
        if binaryMaze(y, x) == 0
            continue;
        end
        
        % 当前节点索引
        currentNode = sub2ind([height, width], y, x);
        
        % 检查四个邻居
        for d = 1:4
            ny = y + directions(d, 1);
            nx = x + directions(d, 2);
            
            % 检查邻居是否在边界内
            if ny >= 1 && ny <= height && nx >= 1 && nx <= width
                % 如果邻居是路径 (不是墙)
                if binaryMaze(ny, nx) == 1
                    % 计算邻居节点索引
                    neighborNode = sub2ind([height, width], ny, nx);
                    
                    % 添加边 (权重为1)
                    count = count + 1;
                    if count > nnz_estimate % 如果预分配的空间不足，扩大数组
                        nnz_estimate = nnz_estimate + numNodes; % 每次增加numNodes个空间
                        row_indices(nnz_estimate) = 0; % 扩大数组
                        col_indices(nnz_estimate) = 0;
                        values(nnz_estimate) = 0;
                    end
                    row_indices(count) = currentNode;
                    col_indices(count) = neighborNode;
                    values(count) = 1;
                end
            end
        end
    end
end

% 将预分配的数组裁剪到实际大小
row_indices = row_indices(1:count);
col_indices = col_indices(1:count);
values = values(1:count);

% 创建稀疏矩阵
distMatrix = sparse(row_indices, col_indices, values, numNodes, numNodes);

% 创建图对象 (可选，Dijkstra算法本身不严格需要，但有助于调试或可视化)
pixelGraph = graph(distMatrix);


% --- Executes on button press in loadMazeButton.
function loadMazeButton_Callback(hObject, eventdata, handles)
% hObject    handle to loadMazeButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Open a file dialog to select a maze image
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.gif', 'Image Files (*.jpg, *.png, *.bmp, *.gif)'}, '选择迷宫图像');
% If the user canceled, return
if isequal(filename, 0) || isequal(pathname, 0)
    return;
end

try
    fullpath = fullfile(pathname, filename);
    [~, ~, ext] = fileparts(filename);
    % Special handling for GIF files
    if strcmpi(ext, '.gif')
        [mazeImage, map] = imread(fullpath, 1); % Read only first frame for GIFs
        if ~isempty(map) % If it's an indexed image with colormap
            mazeImage = ind2rgb(mazeImage, map); % Convert to RGB
        end
    else
        mazeImage = imread(fullpath);
    end
    
    % 将图像转换为灰度图
    if size(mazeImage, 3) == 3
        grayMaze = rgb2gray(mazeImage);
    else
        grayMaze = mazeImage; % 已经是灰度图
    end

    % --- 关键改进：在二值化之前裁剪图像的白色边缘 ---
    % 寻找非白色（非255）像素的边界，假设迷宫墙壁是深色的
    % 如果迷宫是黑墙白路，那么墙壁是0，路径是255。如果外围有白色边，
    % 则需要找到最外层“有内容的”像素。
    
    % 假设迷宫的墙壁是比白色（255）暗的颜色。
    % 找到所有非纯白色的像素（即迷宫内容或墙壁）
    % 注意：这里的阈值 250 需要根据你的实际迷宫图片进行调整。
    % 如果迷宫的墙壁非常亮或者背景色复杂，这个方法可能需要更精细的调整。
    [rows, cols] = find(grayMaze < 250); % 假设250以下的像素表示迷宫内容
    
    if isempty(rows) || isempty(cols)
        set(handles.statusText, 'String', '错误：图像中没有识别到迷宫内容。请检查迷宫图像。');
        return;
    end
    
    % 确定迷宫内容的最小/最大行和列
    minRow = min(rows);
    maxRow = max(rows);
    minCol = min(cols);
    maxCol = max(cols);
    
    % 裁剪图像到迷宫实际内容的边界
    % 额外留出1像素的边框（如果需要），确保墙壁完整
    % 这里的1像素边框是用于确保迷宫最外围的墙体不被切掉，
    % 而不是为了强制白色边缘。
    pad = 1; % 留出1像素的“安全”边距，防止墙壁被切掉
    minRow = max(1, minRow - pad);
    maxRow = min(size(grayMaze, 1), maxRow + pad);
    minCol = max(1, minCol - pad);
    maxCol = min(size(grayMaze, 2), maxCol + pad);

    croppedGrayMaze = grayMaze(minRow:maxRow, minCol:maxCol);
    croppedMazeImage = mazeImage(minRow:maxRow, minCol:maxCol, :); % 裁剪原始彩色图像以保持一致

    % 现在对裁剪后的图像进行二值化
    handles.binaryMaze = imbinarize(croppedGrayMaze);
    handles.mazeImage = croppedMazeImage; % 更新显示的图像，确保GUI显示裁剪后的图像

    % --- 继续执行之前添加的边界强制，这将在裁剪后的图像上生效 ---
    % 这是为了确保裁剪后的迷宫图像最外围的像素依然是墙，
    % 即使裁剪操作没有完美地在墙壁上。
    [h, w] = size(handles.binaryMaze);
    if h > 0 && w > 0
        handles.binaryMaze(1, :) = 0; % 顶部边界
        handles.binaryMaze(h, :) = 0; % 底部边界
        handles.binaryMaze(:, 1) = 0; % 左侧边界
        handles.binaryMaze(:, w) = 0; % 右侧边界
    end
    
    % 存储裁剪后并强制边界的迷宫尺寸
    handles.originalMazeHeight = h;
    handles.originalMazeWidth = w;

    % 现在，使用这个已经裁剪并强制边界的 binaryMaze 构建像素图和距离矩阵
    % buildMazeGraph_with_border 函数内部的边界强制可以保留，作为双重保障，
    % 但主要的裁剪和边界处理已在此处完成。
    [handles.pixelGraph, handles.distanceMatrix] = buildMazeGraph_with_border(handles.binaryMaze);
    
    % 重置起点和终点
    handles.startPoint = [];
    handles.endPoint = [];
    handles.path = [];
    set(handles.startPointLabel, 'String', '起点: 未选择');
    set(handles.endPointLabel, 'String', '终点: 未选择');
    
    % 显示迷宫图像
    axes(handles.mazeAxes);
    imshow(handles.mazeImage); % 显示裁剪后的原始图像
    title('迷宫求解'); % 更新标题以引导用户
    set(handles.statusText, 'String', sprintf('已加载迷宫图像：%s (已裁剪)', filename));
    
catch ME
    set(handles.statusText, 'String', sprintf('错误：无法加载图像 - %s', ME.message));
    fprintf(2, '错误详情：\n'); % 打印错误到命令行，方便调试
    disp(ME); % 显示更详细的错误信息
end
% Update handles structure
guidata(hObject, handles);
% --- Executes on button press in setStartButton.
function setStartButton_Callback(hObject, eventdata, handles)
% hObject    handle to setStartButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Check if we have a maze loaded
if isempty(handles.mazeImage)
    set(handles.statusText, 'String', '错误：请先加载迷宫图像！');
    return;
end

% Set flag to indicate we're waiting for start point selection
handles.waitingForStart = true;
handles.waitingForEnd = false;

% Update status text
set(handles.statusText, 'String', '请在迷宫上点击选择起点');

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in setEndButton.
function setEndButton_Callback(hObject, eventdata, handles)
% hObject    handle to setEndButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isempty(handles.mazeImage)
    set(handles.statusText, 'String', '错误：请先加载迷宫图像！');
    return;
end

handles.waitingForEnd = true;
handles.waitingForStart = false;
set(handles.statusText, 'String', '请在迷宫上点击选择终点');
guidata(hObject, handles);


function clearButton_Callback(hObject, eventdata, handles)
% hObject    handle to clearButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isempty(handles.mazeImage)
    set(handles.statusText, 'String', '请先加载迷宫图像。');
    return;
end

handles.startPoint = [];
handles.endPoint = [];
handles.path = [];
handles.waitingForStart = false;
handles.waitingForEnd = false;
set(handles.startPointLabel, 'String', '起点: 未选择');
set(handles.endPointLabel, 'String', '终点: 未选择');
set(handles.performanceText, 'String', '算法: 未选择,时间: --,内存: --,路径长度: --'); % 清空性能信息

updateMazeDisplay(handles); % 重新显示迷宫，清除路径和点
set(handles.statusText, 'String', '已清除起点、终点和路径');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function mazeAxes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mazeAxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: place code in OpeningFcn to populate mazeAxes


% --- Executes on selection change in algorithmPopup.
function algorithmPopup_Callback(hObject, eventdata, handles)
% hObject    handle to algorithmPopup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns algorithmPopup contents as cell array
%        contents{get(hObject,'Value')} returns selected item from algorithmPopup

% Get selected algorithm index
selection = get(hObject, 'Value');

% Enable/disable heuristic weight field based on algorithm
if selection == 2 % A*
    set(handles.heuristicWeight, 'Enable', 'on');
    set(handles.heuristicWeightLabel, 'Enable', 'on');
else
    set(handles.heuristicWeight, 'Enable', 'off');
    set(handles.heuristicWeightLabel, 'Enable', 'off');
end
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function algorithmPopup_CreateFcn(hObject, eventdata, handles)
% hObject    handle to algorithmPopup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
set(hObject, 'String', {'Dijkstra (经典)', 'A* (启发式)', 'BFS (广度优先)'});


% --- Executes on button press in gpuCheckbox.
function gpuCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to gpuCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of gpuCheckbox
% No need to set handles.useGPU here, it's read directly in solveMazeButton_Callback


% --- Executes on button press in smoothingCheckbox.
function smoothingCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to smoothingCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of smoothingCheckbox
% No need to set anything here, its value is read directly in solveMazeButton_Callback


function heuristicWeight_Callback(hObject, eventdata, handles)
% hObject    handle to heuristicWeight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of heuristicWeight as text
%        str2double(get(hObject,'String')) returns contents of heuristicWeight as a double
% Validate input: must be a positive number
value = str2double(get(hObject, 'String'));
if isnan(value) || value <= 0
    errordlg('启发权重必须是一个正数。', '输入错误');
    set(hObject, 'String', '1.0'); % Reset to default
end


% --- Executes during object creation, after setting all properties.
function heuristicWeight_CreateFcn(hObject, eventdata, handles)
% hObject    handle to heuristicWeight (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
set(hObject, 'String', '1.0'); % Default heuristic weight
