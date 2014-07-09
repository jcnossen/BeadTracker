function txt=generate_motor_script(axis, values, idle_times, motor_wait_time)

    if nargin==0
        axis='magpos';
        values=2:10;
        idle_times=10;
    end
    
    if nargin<4
        motor_wait_time=0.1;
    end
    
    if isscalar(idle_times), 
        idle_times=ones(1,length(values))*idle_times;
    end

    idler = '';
    if motor_wait_time>0, idler = sprintf('idle %.2f; section;', motor_wait_time); end;
    for k=1:length(values)
        lines{k} = sprintf('move %s %.2f; %s idle %.2f; section;\n', ...
            axis, values(k), idler, idle_times(k));
    end
    txt = cell2mat(lines);

    clipboard('copy', txt);

end