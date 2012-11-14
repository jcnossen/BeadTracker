% Generate offset/bias/scatter values from LVDatasetTest.vi output
function error_values_plot(dirname)

    if nargin==0
        figure(1); scatter_plot('dataset_result/roi80');    
        figure(2); scatter_plot('dataset_result/roi400');
    end
    
end

function scatter_plot(dirname)
    truepos = dlmread([dirname '/true-pos.csv']);
    jtrkcom = dlmread([dirname '/jtrk-com.csv']);
    jtrkxcor = dlmread([dirname '/jtrk-xcor.csv']);
    jtrkqi = dlmread([dirname '/jtrk-qi.csv']);
    lvtrkcom = dlmread([dirname '/lvtrk-com.csv']);
    lvtrkqi = dlmread([dirname '/lvtrk-qi.csv']);
    lvtrkxcor = dlmread([dirname '/lvtrk-xcor.csv']);
    z_results = dlmread([dirname '/measured-z.csv']);

%    plot ( truepos-jtrkcom );

    % COM
    subplot(421);
    scatter(truepos(:,2),lvtrkcom(:,2)-truepos(:,2),'.'); title('LV COM');
    subplot(422);
    scatter(truepos(:,2),jtrkcom(:,2)-truepos(:,2),'.'); title('C++ COM');

    % XCor
    subplot(423);
    scatter(truepos(:,2),lvtrkxcor(:,2)-truepos(:,2)-0.5,'.'); title('LV XCor ( -0.5 offset )');
    subplot(424);
    scatter(truepos(:,2),jtrkxcor(:,2)-truepos(:,2),'.'); title('C++ XCor');

    % QI
    subplot(425);
    scatter(truepos(:,2),lvtrkqi(:,2)-truepos(:,2),'.'); title('LV QI');
    subplot(426);
    scatter(truepos(:,2),jtrkqi(:,2)-truepos(:,2),'.'); title('C++ QI');

    % Z
    subplot(427);
    scatter(truepos(:,2),z_results(:,2)-truepos(:,3),'.'); title('LV Z');
    subplot(428);
    scatter(truepos(:,2),z_results(:,1)-truepos(:,3),'.'); title('C++ Z');
    
end


