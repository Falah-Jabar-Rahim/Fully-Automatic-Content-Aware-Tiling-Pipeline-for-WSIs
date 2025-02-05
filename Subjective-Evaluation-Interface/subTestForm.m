function varargout = subTestForm(varargin)
%% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @subTestForm_OpeningFcn, ...
    'gui_OutputFcn',  @subTestForm_OutputFcn, ...
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
%  Begin initialization code - DO NOT EDIT
function subTestForm_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Initialize the progress bar at the bottom of the GUI
    handles.progressBar = uicontrol('Style', 'text', ...
                                    'Position', [10, 10, 1420, 20], ...
                                    'String', 'Progress: 0%', ...
                                    'HorizontalAlignment', 'left', ...
                                    'BackgroundColor', [0.95, 0.95, 0.95], ...
                                    'ForegroundColor', 'blue');
    
handles.progress = 0;  % Initialize progress to 0%
handles.output = hObject;
guidata(hObject, handles);  % Update handles structur
%%

% set inputs parameters  
filename = 'Tiles.xlsx';
handles.Tc =1;
handles.start=1;
handles.num_train=3; % number of training examples 




imageData = readtable(filename);
data = cell(1, height(imageData));
% Loop through each row of the table
for i = 1:height(imageData)
    % Extract the filenames from the table row
    centerImage = imageData.Center{i};
    topImage = imageData.Top{i};
    leftImage = imageData.Left{i};
    bottomImage = imageData.Down{i};
    rightImage = imageData.Right{i};
    % Create the 2x5 cell matrix for this row
    imageSet = {centerImage, topImage, leftImage, bottomImage, rightImage; [], [], [], [], []};
    % Store the matrix in the cell array
    data{i} = imageSet;
end

%% output results 
result= cell(length(data)+1,2);
result{1,1}='Sets';
result{1,2}='Selected';

%% Read  image sets
handles.DataLen = length(data);
row = length(data);
col = length(data{1,1}(1,:));

for i=1:row
    for j=1:col
        seq_name = data{1,i}(1,j);
        I = imread(seq_name{1,1});
        temp = cell(1);
        temp{1,1} = I;
        data{1,i}(2,j) = temp;
        s = strcat('Loading ', seq_name ,' sequence');
        hand = waitbar(0,s);
        close(hand);
    end
end

handles.data = data;
handles.result = result;
handles.img_num =col;
handles.img_sets =row;
%% Initialize the UI
set(handles.uipanel2, 'Visible', 'on');
set(handles.axes1, 'Visible', 'off');
set(handles.axes2, 'Visible', 'off');
set(handles.axes3, 'Visible', 'off');
set(handles.axes4, 'Visible', 'off');
set(handles.axes5, 'Visible', 'off');
set(handles.finish, 'Visible', 'off');
set(handles.pushbutton10,'Visible', 'off')
set(handles.pushbutton12,'Visible', 'off')
set(handles.pushbutton13,'Visible', 'off')
 set(handles.pushbutton14,'Visible', 'off')
 set(handles.pushbutton15,'Visible', 'off')
 set(handles.pushbutton16,'Visible', 'off')
set(handles.progressBar,'Visible', 'off')
set(handles.pushbutton11,'Visible', 'off')
set(handles.uipanel8,'Visible', 'off')
set(handles.pushbutton11,'Value', 0)
set(handles.pushbutton12,'Value', 0)
set(handles.pushbutton13,'Value', 0)
set(handles.pushbutton14,'Value', 0)
set(handles.pushbutton15,'Value', 0)
set(handles.pushbutton16,'Value', 0)
 set(handles.text45,'Visible','off');



% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton11.
function selection_Callback(hObject, eventdata, handles) 

Tc = handles.Tc;
handles.result{Tc,1} = Tc;

% Calculate the progress percentage
    progress = (Tc / handles.img_sets) * 100;
    handles.progress = progress;  % Update the handles with the new progress
    
    % Update the progress bar text
    set(handles.progressBar, 'String', sprintf('Progress: %.0f%%', progress));
    
    % Optionally, update the position of the bar or its color
    bar_width = 1420 * (progress / 100);  % Adjust the bar width based on progress
    set(handles.progressBar, 'Position', [10, 10, bar_width, 20]);
     set(handles.progressBar, 'BackgroundColor', [0.8, 0.8, 0.8]);

   

if get(handles.pushbutton11,'Value') ==1
handles.result{Tc,2} = handles.center_img_name;

elseif get(handles.pushbutton12,'Value') ==1
 handles.result{Tc,2} =handles.top_img_name;

elseif get(handles.pushbutton13,'Value') ==1
 handles.result{Tc,2} =handles.left_img_name;
elseif get(handles.pushbutton14,'Value') ==1
 handles.result{Tc,2} =handles.bottom_img_name;
elseif get(handles.pushbutton15,'Value') ==1
 handles.result{Tc,2} =handles.right_img_name;

else
 handles.result{handles.Tc,2} ="None";

end


if Tc == handles.img_sets

    result=handles.result;
    filename = handles.filename;
    % result{Tc+1,1}='Age';
    % result(Tc+1,2)=num2cell(handles.fileage);
    writecell(result,strcat(filename,".xlsx") );

    cla(handles.axes1);
    cla(handles.axes2);
    cla(handles.axes3);
    cla(handles.axes4);
    cla(handles.axes5);

    set(handles.pushbutton11,'Visible', 'off')
    set(handles.pushbutton12,'Visible', 'off')
    set(handles.pushbutton13,'Visible', 'off')
    set(handles.pushbutton14,'Visible', 'off')
    set(handles.pushbutton10,'Visible','off');
    set(handles.pushbutton15,'Visible', 'off')
    set(handles.pushbutton16,'Visible', 'off')
    set(handles.finish, 'Visible', 'on');
pause(5);
close all force;

else
    if Tc == handles.num_train
 
    set(handles.pushbutton11,'Value', 0)
    set(handles.pushbutton12,'Value', 0)
    set(handles.pushbutton13,'Value', 0)
    set(handles.pushbutton14,'Value', 0)
    set(handles.pushbutton15,'Value', 0)
    set(handles.pushbutton16,'Value', 0)
    set(handles.axes1, 'Visible', 'off');
    set(handles.axes2, 'Visible', 'off');
    set(handles.axes3, 'Visible', 'off');
    set(handles.axes4, 'Visible', 'off');
    set(handles.axes5, 'Visible', 'off');
    set(handles.pushbutton11,'Visible','off');
    set(handles.pushbutton12,'Visible','off');
    set(handles.pushbutton13,'Visible','off');
    set(handles.pushbutton14,'Visible','off');
    set(handles.pushbutton15,'Visible','off');
    set(handles.pushbutton16,'Visible','off');
    set(handles.pushbutton10,'Visible','off');
    cla(handles.axes1);
    cla(handles.axes2);
    cla(handles.axes3);
    cla(handles.axes4);
    cla(handles.axes5);
    handles.Tc = Tc + 1;

   set(handles.text45, 'Visible', 'on');

% Countdown loop
    for t = 10:-1:0
        set(handles.text45, 'String', sprintf('Testing tasks starts in %d', t));
        pause(1);  % Wait for 1 second
    end

   set(handles.pushbutton10,'Visible','on');
   set(handles.text45, 'Visible', 'off');


    else


    set(handles.pushbutton11,'Value', 0)
    set(handles.pushbutton12,'Value', 0)
    set(handles.pushbutton13,'Value', 0)
    set(handles.pushbutton14,'Value', 0)
    set(handles.pushbutton15,'Value', 0)
    set(handles.pushbutton16,'Value', 0)
    set(handles.axes1, 'Visible', 'off');
    set(handles.axes2, 'Visible', 'off');
    set(handles.axes3, 'Visible', 'off');
    set(handles.axes4, 'Visible', 'off');
    set(handles.axes5, 'Visible', 'off');
    set(handles.pushbutton11,'Visible','off');
    set(handles.pushbutton12,'Visible','off');
    set(handles.pushbutton13,'Visible','off');
    set(handles.pushbutton14,'Visible','off');
    set(handles.pushbutton15,'Visible','off');
    set(handles.pushbutton16,'Visible','off');
    set(handles.pushbutton10,'Visible','on');
    cla(handles.axes1);
    cla(handles.axes2);
    cla(handles.axes3);
    cla(handles.axes4);
    cla(handles.axes5);
    handles.Tc = Tc + 1;

    end

guidata(hObject, handles);

end

% --- Executes on button press in pushbutton18.
function pushbutton18_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.start==1
    set(handles.uipanel2, 'Visible', 'off');
    filename = get(handles.nameBox, 'String');
    handles.filename = filename;
    handles.fileage= str2double(handles.ageBox.String); %returns contents of ageBox as a double
    set(handles.uipanel8,'Visible', 'on')
    handles.start=0;
    guidata(hObject, handles);
else

    set(handles.pushbutton11,'Visible', 'on')
    set(handles.pushbutton12,'Visible', 'on')
    set(handles.pushbutton13,'Visible', 'on')
    set(handles.pushbutton14,'Visible', 'on')
    set(handles.pushbutton10,'Visible','off');
    set(handles.pushbutton15,'Visible', 'on')
    set(handles.pushbutton16,'Visible', 'on')
    set(handles.progressBar,'Visible', 'on')
    set(handles.uipanel8,'Visible', 'off')

    
    Tc = handles.Tc;
    data = handles.data;
    
    img_set = data{1,Tc};
    
    handles.center_img_name= img_set{1,1};
    center_img = img_set{2,1};
    
    handles.top_img_name= img_set{1,2};
    top_img = img_set{2,2};
    
    handles.left_img_name= img_set{1,3};
    left_img = img_set{2,3};
    
    handles.bottom_img_name= img_set{1,4};
    bottom_img = img_set{2,4};
    
    handles.right_img_name= img_set{1,5};
    right_img = img_set{2,5};
    
    imshow(center_img,'Parent',handles.axes1);
    imshow(top_img,'Parent',handles.axes2);
    imshow(left_img,'Parent',handles.axes3);
    imshow(bottom_img,'Parent',handles.axes4);
    imshow(right_img,'Parent',handles.axes5);
    % Update handles structure
    guidata(hObject, handles);

end

%% Outputs from this function are returned to the command line.
function varargout = subTestForm_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Get default command line output from handles structure
varargout{1} = handles.output;

%% Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
pushbutton9_Callback(hObject, eventdata, handles)

function ageBox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ageBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function nameBox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nameBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function nameBox_Callback(hObject, eventdata, handles)
% hObject    handle to nameBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function ageBox_Callback(hObject, eventdata, handles)
% hObject    handle to ageBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%%

%%


% --- Executes on button press in pushbutton19.
function pushbutton19_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
