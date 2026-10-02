% Replot the saved NAV x/y tube projections from the adjacent compact CSVs.
% Standard native is new (2026-10-02); the three GPU methods and robust
% native are historical. The old "ours" GPU engine is not current working P3.
root = fileparts(mfilename('fullpath'));
base = fileparts(root);
folders = {'nav_standard_fourway_saved_20261002', 'nav_robust_fourway_saved_20261002'};
titles = {'Standard', 'Robust'};
states = {'x', 'y'};
methods = {'ours', 'huan', 'xiangru', 'flowstar_native'};
labels = {'old GPU ours', 'old Huan', 'old Xiangru', 'Flow* native'};
colors = [0.000 0.447 0.698; 0.902 0.624 0.000; 0.000 0.620 0.451; 0.800 0.475 0.655];
styles = {'-', '--', ':', '-.'};
figure('Color', 'w', 'Position', [100, 100, 1200, 820]);
for suite = 1:2
    data = readtable(fullfile(base, folders{suite}, 'xy_saved_curves.csv'));
    for axis_id = 1:2
        subplot(2, 2, (suite - 1) * 2 + axis_id);
        hold on;
        for method_id = 1:4
            take = strcmp(data.method, methods{method_id}) & strcmp(data.state, states{axis_id});
            [t, order] = sort(data.t_end(take));
            lower = data.tube_lo(take); lower = lower(order);
            upper = data.tube_hi(take); upper = upper(order);
            patch([t; flipud(t)], [lower; flipud(upper)], colors(method_id,:), ...
                  'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(t, lower, 'Color', colors(method_id,:), 'LineStyle', styles{method_id}, ...
                 'LineWidth', 0.8, 'DisplayName', labels{method_id});
            plot(t, upper, 'Color', colors(method_id,:), 'LineStyle', styles{method_id}, ...
                 'LineWidth', 0.8, 'HandleVisibility', 'off');
        end
        title([titles{suite} ': ' states{axis_id}]);
        xlabel('Time (s)'); ylabel([states{axis_id} ' saved tube']);
        xlim([0, 6]); grid on;
        if suite == 1 && axis_id == 1, legend('Location', 'best'); end
    end
end
sgtitle('NAV saved flowpipe x/y projections: historical GPU versus native');
% x or y alone cannot establish avoidance of the joint obstacle [1,2]^2.
% The separate per-box scan checks that joint intersection on saved tubes.
print(gcf, fullfile(root, 'nav_saved_xy_tubes_matlab.png'), '-dpng', '-r200');
