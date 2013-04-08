function plot_speeds()
    s690=dlmread('speeds-geforce690.txt');
    s560=dlmread('speeds-geforce560.txt');
    s560sm=dlmread('speeds-geforce560-sm.txt', '\t', 1, 0);
    
    s = [ s690(:,[1 3]) s560(:,3) s560sm(:,[1 3]) ];

    figure();
    x = 40 + ((0:11)*10);
    plot(x,s);
    title('Tracking speed vs ROI size');
    xlabel('ROI size [pixels]');
    
    legend( '12 core CPU', 'Geforce 690', 'Geforce 560', '4 core cpu', 'Geforce 560 (bad parallel code)');
