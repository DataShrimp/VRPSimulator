import java.util.ArrayList;

public class TSPSolver {
    private ArrayList<Integer> outputArray = new ArrayList<Integer>();
    private double g[][], d[][];
    private int p[][], npow, N;
    public static long time;
    public static double shortest;

    public TSPSolver() {}

    public ArrayList<Integer> computeTSP(double[][] inputArray) {
        long start = System.nanoTime();

        N = inputArray.length;
        // 使用2^N表示剩余目标地集合，如N=3，0表示000空集，6表示110即{2,1}目的地集合
        npow = (int)Math.pow(2, N);
        // 记录动态规划中间结果
        g = new double[N][npow];
        // 记录路程信息
        p = new int[N][npow];
        d = inputArray;

        for (int i=0; i<N; i++) {
            for (int j=0; j<npow; j++) {
                g[i][j] = -1;
                p[i][j] = -1;
            }
        }

        // 设置初始距离
        for (int i=0; i<N; i++) {
            g[i][0] = inputArray[i][0];
        }
        // 迭代计算从{0}到目的地{1,2,...,N}的最短路径
        shortest = tsp(0, npow-2);
        // 获得最短路径
        outputArray.add(0);
        getPath(0, npow-2);
        outputArray.add(0);

        long end = System.nanoTime();
        time = (end - start)/1000;
        return outputArray;
    }

    private double tsp(int start, int set) {
        int masked, mask;
        double result = -1, temp;
        if (g[start][set] > 0) {
            return g[start][set];
        } else {
            for (int i=0; i<N; i++) {
                mask = npow - 1 - (int)Math.pow(2,i);
                masked = set & mask;
                if (masked != set) {
                    temp = d[start][i] + tsp(i, masked);
                    // 未访问过或者更短的路径
                    if (result < 0 || result > temp) {
                        result = temp;
                        p[start][set] = i;
                    }
                }
            }
            g[start][set] = result;
            return result;
        }
    }

    private void getPath(int start, int set) {
        if (p[start][set] == -1) {
            return;
        }

        int x = p[start][set];
        int mask = npow - 1 - (int)Math.pow(2, x);
        int masked = set & mask;

        outputArray.add(x);
        getPath(x, masked);
    }
}
