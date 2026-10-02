% Saved Single Pendulum two-state boxes; no solver or property checker is run.
% Generated from saved_tubes.csv. This example has not been run in MATLAB.
T = readtable(fullfile(fileparts(mfilename('fullpath')), 'saved_tubes.csv'));
methods = {'P3', 'Huan', 'Xiangru', 'Flow* native'};
states = {'x1', 'x2'};
colors = [0 114 178; 213 94 0; 204 121 167; 0 158 115] / 255;
figure('Color', 'w');
for d = 1:2
    subplot(2,2,d); hold on; grid on;
    if d == 1
        patch([0 1 1 0], [0 0 1 1], [0.87 0.93 0.87], ...
              'FaceAlpha', .5, 'EdgeColor', 'none');
        xline(.5, ':', 'Color', [.5 .5 .5]);
    end
    for m = 1:4
        R = T(strcmp(T.method, methods{m}) & strcmp(T.state, states{d}), :);
        assert(height(R) == 100 && all(R.step == (1:100)'));
        style = '-'; if m == 3, style = '--'; end
        stairs([0; R.t_end], [R.tube_lo; R.tube_lo(end)], ...
               'Color', colors(m,:), 'LineStyle', style, 'DisplayName', methods{m});
        stairs([0; R.t_end], [R.tube_hi; R.tube_hi(end)], ...
               'Color', colors(m,:), 'LineStyle', style, 'HandleVisibility', 'off');
    end
    xlim([0 1]); xlabel('time (s)'); ylabel(states{d});
    title([states{d} ' saved whole-step tube']);
    if d == 1, legend('Location', 'best'); end
    subplot(2,2,d+2); hold on; grid on;
    H = T(strcmp(T.method, 'Huan') & strcmp(T.state, states{d}), :);
    for m = 1:4
        R = T(strcmp(T.method, methods{m}) & strcmp(T.state, states{d}), :);
        stairs(R.t_end, R.endpoint_lo-H.endpoint_lo, 'Color', colors(m,:));
        stairs(R.t_end, R.endpoint_hi-H.endpoint_hi, ...
               'Color', colors(m,:), 'LineStyle', '--');
    end
    xlim([.8 1]); xlabel('time (s)'); ylabel(['delta ' states{d}]);
    title([states{d} ' endpoint minus Huan (lower solid / upper dashed)']);
end
sgtitle('Single Pendulum, named two-physical-state profile');
