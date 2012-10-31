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
    lvtrkcom = dlmread([dirname '/lvtrk-com.csv']);
    lvtrkqi = dlmread([dirname '/lvtrk-qi.csv']);
    lvtrkxcor = dlmread([dirname '/lvtrk-xcor.csv']);

%    plot ( truepos-jtrkcom );

    subplot(311);
    scatter(truepos(:,2),jtrkxcor(:,2)-truepos(:,2),'.'); title('C++ XCor Interpolated');
    subplot(312);
    scatter(truepos(:,2),lvtrkxcor(:,2)-truepos(:,2)-0.5,'.'); title('LV XCor ( -0.5 offset )');
    subplot(313);
    scatter(truepos(:,2),lvtrkqi(:,2)-truepos(:,2),'.'); title('QI');

end


