function plot_speeds()
    s690=dlmread('speeds-geforce690.txt');
    s560=dlmread('speeds-geforce560.txt');
    
    s = [ s690(:,[1 3]) s560(:,[1 3]) ];

    figure();
    x = 40 + ((0:11)*10);
    plot(x,s);
    title('Tracking speed vs ROI size');
    xlabel('ROI size [pixels]');
    
    legend( '12 core CPU', 'Geforce 690', '4 core CPU', 'Geforce 560');
    