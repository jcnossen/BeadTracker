% Generate offset/bias/scatter values from LVDatasetTest.vi output
function error_values_plot(dirname)

    if nargin==0
        %scatter_bias_plot('dataset_result/roi80');    
        scatter_bias_plot('dataset_result/roi400');
    end
end

function scatter_bias_plot(dirname)

    xbinsize = 250;

    truepos = dlmread([dirname '/true-pos.csv']);
    jtrkcom = dlmread([dirname '/jtrk-com.csv']);
    jtrkxcor = dlmread([dirname '/jtrk-xcor.csv']);
    jtrkqi = dlmread([dirname '/jtrk-qi.csv']);
    lvtrkcom = dlmread([dirname '/lvtrk-com.csv']);
    lvtrkqi = dlmread([dirname '/lvtrk-qi.csv']);
    lvtrkxcor = dlmread([dirname '/lvtrk-xcor.csv']);
    z_results = dlmread([dirname '/measured-z.csv']);

%    plot ( truepos-jtrkcom );
    r.y = sb(xbinsize, truepos(:,2),0);
    r.lvcom = sb(xbinsize, lvtrkcom(:,2)-truepos(:,2) );
    r.cppcom = sb(xbinsize, jtrkcom(:,2)-truepos(:,2) );
    r.lvxcor = sb(xbinsize, lvtrkxcor(:,2)-truepos(:,2)-0.5);
    r.cppxcor = sb(xbinsize, jtrkxcor(:,2)-truepos(:,2));
    r.lvqi = sb(xbinsize, lvtrkqi(:,2)-truepos(:,2));
    r.cppqi = sb(xbinsize, jtrkqi(:,2)-truepos(:,2));
    
%    plot(r.y(:,1), [ r.cppcom r.cppxcor r.cppqi ]);
%    legend('COM (bias)', 'COM (scatter)', 'Cross-Cor (bias)','Cross-Cor(scatter)', 'QI (bias)', 'QI (scatter)');
%    plot(r.y(:,1), [ r.lvcom r.lvxcor r.lvqi ]);
%    legend('COM (bias)', 'COM (scatter)', 'Cross-Cor (bias)','Cross-Cor(scatter)', 'QI (bias)', 'QI (scatter)');

    plots = { r.lvxcor r.lvqi r.cppxcor r.cppqi };

    colors = {'r', 'g', 'b', 'x' };
    for u=1:length(plots)
        data = plots{u};
        h(u) = semilogy(r.y(:,1),data(:,1) , 'LineWidth', 2);
    end
    for u=1:4
        set(h(u), 'Color', colors{u});
        set(h(u), 'LineStyle', ':');
    end
    legend('XCor (bias)','XCor(scatter)', 'QI (bias)', 'QI (scatter)');%, 'C++ XCor (bias)','C++ XCor(scatter)', 'C++ QI (bias)', 'C++ QI (scatter)');
%    plot(r.y(:,1), [ r.cppcom r.cppxcor r.cppqi ]);
    xlim([-2 0]);
    
    title('Scatter/Bias plot');
    
    %figure(2);
    %scatterplots();

    % COM
    function scatterplots()
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
    
end

function result = sb(bsize, d, makeabs)
    numbins = size(d,1)/bsize;
    %fprintf('numbins: %d\n',numbins);
    result = zeros(numbins, 2);
    if nargin<3
        makeabs=1;
    end
    for k=1:numbins
        bin = (1:bsize) + (k-1) * bsize;
        if makeabs
            bind = abs(d(bin,:));
        else
            bind = d(bin,:);
        end
        
        result(k, :) = [ mean(bind) std(bind) ];
    end
end
