% Plot saved four-method DP less tube unions. This script runs no solver.
% Requires tube_union.csv in the same directory; generated here, not run in MATLAB.
T = readtable(fullfile(fileparts(mfilename('fullpath')), 'tube_union.csv'));
methods = {'native', 'Huan', 'Xiangru', 'ours/P3'};
states = {'theta1', 'theta2', 'theta1_dot', 'theta2_dot'};
colors = [31 119 180; 217 95 2; 117 112 179; 27 158 119] / 255;
figure('Color', 'w');
for d = 1:4
    subplot(2, 2, d); hold on; grid on;
    V = T(strcmp(T.state, states{d}), :);
    yBounds = [min(V.tube_lo), max(V.tube_hi)];
    pad = 0.05 * (yBounds(2) - yBounds(1));
    ylim([yBounds(1)-pad, yBounds(2)+pad]);
    patch([0 1 1 0], [-1.7 -1.7 2 2], [0.85 0.93 0.86], ...
          'FaceAlpha', 0.55, 'EdgeColor', 'none', ...
          'DisplayName', 'Safe [-1.7, 2]');
    for m = 1:4
        R = T(strcmp(T.method, methods{m}) & strcmp(T.state, states{d}), :);
        assert(height(R) == 100 && all(R.step == (1:100)'));
        x = [0; R.t_end]; lo = [R.tube_lo; R.tube_lo(end)];
        hi = [R.tube_hi; R.tube_hi(end)];
        lineStyle = '-'; if m == 3, lineStyle = '--'; end
        stairs(x, lo, 'Color', colors(m, :), 'LineStyle', lineStyle, ...
               'DisplayName', methods{m});
        stairs(x, hi, 'Color', colors(m, :), 'LineStyle', lineStyle, ...
               'HandleVisibility', 'off');
    end
    plot([0 0], [1 1.3], 'ko-', 'MarkerSize', 3, 'LineWidth', 2, ...
         'DisplayName', 'Initial [1, 1.3] at t=0');
    title([states{d} ' | Safe [-1.7, 2] (cropped)']);
    xlabel('t (s)'); ylabel('state interval'); xlim([0 1]);
    if d == 1, legend('Location', 'best'); end
end
sgtitle('Double Pendulum less robust: saved 225-box tube unions');
