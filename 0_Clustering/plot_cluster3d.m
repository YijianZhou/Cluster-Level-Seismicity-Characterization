% Read and visualize earthquake clusters in 3D

% Define spatial constraints
dep_rng = [-1, 11];
% Set the starting cluster ID to resume work
start_cluster_id = 0; % Change this to the desired starting ID

% Open file
%fid = fopen('output/db-seis_3-merged-cluster.csv', 'r');
fid = fopen('output/cluster-60_3.csv', 'r');
if fid == -1
    error('Could not open file.');
end

% Process clusters one at a time
cluster_id = 0;
events = [];

while ~feof(fid)
    line = strtrim(fgetl(fid));
    if startsWith(line, '#')
    
        % If there are events from the previous cluster, plot them
        if ~isempty(events)
            fprintf('Plotting Cluster %d with %d events\n', cluster_id, size(events, 1));
    
            % Convert data for easy access
            lat = cell2mat(events(:, 2));
            lon = cell2mat(events(:, 3));
            dep = -cell2mat(events(:, 4));
            mag = 1.5 * (cell2mat(events(:, 5)) + 2);

            % Convert origin time (ISO 8601) to datenum
            ot_str = events(:, 1);  % Cell array of datetime strings
            ot_num = datenum(ot_str, 'yyyy-mm-ddTHH:MM:SS.FFFZ');  % Convert to MATLAB datenum
    
            % Compute relative time in float days
            rel_time = ot_num - min(ot_num);
            ot_min_str = datestr(min(ot_num), 'yyyy-mm-dd HH:MM:SS');

            % Normalize relative time for colormap scaling
            norm_time = (rel_time - min(rel_time)) / (max(rel_time) - min(rel_time));

            % Use jet colormap for temporal coloring
            cmap = jet;
            colors = interp1(linspace(0, 1, size(cmap, 1)), cmap, norm_time);
    
            % Plot earthquakes in 3D
            figure;
            %figure('Color', [.4 .4 .4]); % Set background to dark gray
            %set(gca, 'Color', [.2 .2 .2]); % Set axes background to dark gray
            scatter3(lon, lat, dep, mag, colors, 'filled', 'MarkerEdgeColor', 'none');
            hold on;
    
            % Add colorbar and labels
            c = colorbar;
            colormap('turbo');
            caxis([min(rel_time) max(rel_time)]);  % Set color axis in relative float days
            ylabel(c, sprintf('Relative Time since %s (days)', ot_min_str), 'FontSize', 12);
    
            xlabel('Longitude');
            ylabel('Latitude');
            zlabel('Depth (km)');
            title(sprintf('Cluster %d', cluster_id));
            grid on;
            
            % Enable brushing for selection
            brush on;
            subcluster_id = 1;
            while true
                pause;  % Allow user to manually brush-select points and fine-tune selection
                brushedData = findobj(gca, 'Type', 'Scatter');
                if isempty(brushedData)
                    break;
                end
                brushedData = brushedData(1); % Ensure only one scatter object is used
                selected_indices = find(brushedData.BrushData);
                
                if isempty(selected_indices)
                    continue; % Wait for next selection instead of breaking immediately
                end
                
                % Extract selected subcluster
                selected_events = events(selected_indices, :);
                output_filename = sprintf('output/cluster-%d_%d.csv', cluster_id, subcluster_id);
                writetable(cell2table(selected_events), output_filename, 'WriteVariableNames', false);
                fprintf('Saved selected events to %s\n', output_filename);
                
                % Remove selected points from current figure
                lon(selected_indices) = [];
                lat(selected_indices) = [];
                dep(selected_indices) = [];
                mag(selected_indices) = [];
                events(selected_indices, :) = [];
                
                % Update plot to reflect removed points
                set(brushedData, 'XData', lon, 'YData', lat, 'ZData', dep, 'SizeData', mag, 'CData', colors(1:length(lon), :));
                brushedData.BrushData = zeros(size(brushedData.BrushData));
                subcluster_id = subcluster_id + 1;
            end
            
            close;
            fprintf('Finished processing Cluster %d\n', cluster_id);
        end
        
        % Start a new cluster
        cluster_id = cluster_id + 1;
        fprintf('Processing Cluster %d\n', cluster_id);
        if cluster_id < start_cluster_id
            fprintf('Skipping Cluster %d (below start ID)\n', cluster_id);
            events = [];
            continue;
        end
        events = [];
        continue;
        
    elseif ~isempty(line) && cluster_id >= start_cluster_id
        codes = split(line, ',');
        ot = codes{1};                      % Origin time as string
        lat = str2double(codes{2});         % Latitude as float
        lon = str2double(codes{3});         % Longitude as float
        dep = str2double(codes{4});         % Depth as float
        mag = str2double(codes{5});         % Magnitude as float
        evid = str2double(codes{6});        % Event ID as integer
        events = [events; {ot, lat, lon, dep, mag, evid}];
    end
end

fclose(fid);
fprintf('All clusters processed.\n');
