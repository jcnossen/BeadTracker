% Generate offset/bias/scatter values from LVDatasetTest.vi output
function error_values_plot(dirname)

    if nargin==0
        dirname = 'dataset_result/roi80';
    end

    truepos = dlmread([dirname '/true-pos.csv']);
    jtrkcom = dlmread([dirname '/jtrk-com.csv']);
    jtrkxcor = dlmread([dirname '/jtrk-xcor.csv']);
    lvtrkcom = dlmread([dirname '/lvtrk-com.csv']);
    lvtrkqi = dlmread([dirname '/lvtrk-qi.csv']);
    lvtrkxcor = dlmread([dirname '/lvtrk-xcor.csv']);

%    plot ( truepos-jtrkcom );

figure(1);
    scatter(truepos(:,2),jtrkxcor(:,2)-truepos(:,2));
    figure(2);
    scatter(truepos(:,2),lvtrkxcor(:,2)-truepos(:,2));
    figure(3); scatter(truepos(:,2),lvtrkqi(:,2)-truepos(:,2));
    
end


