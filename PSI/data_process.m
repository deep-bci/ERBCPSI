function Data = Copy_of_data_process(file_path, Fs, Time_start, Time_end, class)
%Description: Preprocess of DEAP dataset, including 4 steps:
%1. load data and select the data within [Time_start,Time_end]s;

mat = load(file_path);
raw_data = mat.data;
labels = mat.label;

% Time window selection
N = 62;
[M, ~, P] = size(raw_data);
nSegment = numel(Time_start);
T1 = Time_start * Fs;
T2 = Time_end * Fs;

% Segmention and detrend
for iseg = 1:nSegment
    SamplePoint = T1(iseg)+1:T2(iseg);
    for m = 1: M
         Temp  = zeros(N, numel(SamplePoint));
        for n = 1:N
            Temp(n,:) = raw_data(m,n,SamplePoint);   
        end
        Data{m}.Segment{iseg} = Temp;
    end
    clear Temp
    
    %Filter
    wdelta_1 = 0.5/(Fs/2);
    wtheta_1 = 4/(Fs/2);
    walpha_1 = 8/(Fs/2);
    wbeta_1 = 12/(Fs/2);
    wbeta_2 = 30/(Fs/2);
    wgamma1 = 45/(Fs/2);
             
    bdelta = fir1(62,[wdelta_1 wtheta_1]);
    btheta = fir1(62,[wtheta_1 walpha_1]);
    balpha = fir1(62,[walpha_1 wbeta_1]);
    bbeta = fir1(62,[wbeta_1 wbeta_2]);
    bgamma = fir1(62,[wbeta_2 wgamma1]);


    M = numel(Data);
    for m = 1:M
        Xs = Data{m}.Segment{iseg};
        Data{m}.Delta{iseg} = filtfilt(bdelta,1,Xs')';
        Data{m}.Delta{iseg} = phase_synchronization(Data{m}.Delta{iseg});
        Data{m}.Theta{iseg} = filtfilt(btheta,1,Xs')';
        Data{m}.Theta{iseg} = phase_synchronization(Data{m}.Theta{iseg});
        Data{m}.Alpha{iseg} = filtfilt(balpha,1,Xs')';
        Data{m}.Alpha{iseg} = phase_synchronization(Data{m}.Alpha{iseg});
        Data{m}.Beta{iseg} = filtfilt(bbeta,1,Xs')';
        Data{m}.Beta{iseg} = phase_synchronization(Data{m}.Beta{iseg});
        Data{m}.Gamma{iseg} = filtfilt(bgamma,1,Xs')';
        Data{m}.Gamma{iseg} = phase_synchronization(Data{m}.Gamma{iseg});
    end
end
% 
% if class == 2
%    for i = 1:M
%       if raw_labels(i, 1) <= 5
%          labels{i}.arousal{1} = 0; 
%       else raw_labels(i, 1) > 5
%           labels{i}.arousal{1} = 1;
%       end   
%    end
% if class == 2
%    for i = 1:M
%       if raw_labels(i,1) <= 5
%          labels{i}.valence{1} = 0; 
%       else raw_labels(i,1) > 5
%           labels{i}.valence{1} = 1;
%       end
%       if raw_labels(i,2) <= 5
%          labels{i}.arousal{1} = 0; 
%       else raw_labels(i,2) > 5
%          labels{i}.arousal{1} = 1;
%       end     
%    end
% else class == 4
%     for i = 1:M
%         if (raw_labels(i,1) <=5) && (raw_labels(i,2) <=5)
%             labels{i} = 0;
%         elseif(raw_labels(i,1) > 5) && (raw_labels(i,2) <=5)
%             labels{i} = 1;
%         elseif(raw_labels(i,1) <= 5) && (raw_labels(i,2) >5)
%             labels{i} = 2;
%         else(raw_labels(i,1) > 5) && (raw_labels(i,2) > 5)
%             labels{i} = 3;
%         end
%     end
% end
trail = strsplit(file_path, '\');
trail = trail(6);
% save_path = strcat('D:\Matlab\Matlab-Learn-master\newnew\filtrate_4.8_5.2_deap_2_class_valence\',trail);
save_path = strcat('D:\DataSets\SEED\2min_data_15trial\ThirdWeek_2min_data_PSI\', trail);
save_path = cell2mat(save_path);
save(save_path, 'Data', 'labels');
disp(save_path);

