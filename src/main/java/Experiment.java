import java.util.ArrayList;

public class Experiment {
    private int index = 0;
    private double distance = 0.0;

    private ArrayList<Integer> indexList = null;
    public static int N = 0;

    // 用于TSP求解计算使用
    private TSPSolver solver = null;
    private double optDist = 0.0;
    private ArrayList<Integer> optIndexList = null;
    private double[][] cities;
    private double[][] graph;

    public Experiment() {}

    public Experiment(int n) {
        this.initExp(n);
    }

    public void initExp(int n) {
        indexList = new ArrayList<>();
        indexList.add(0);
        for (int i=0; i<n; i++) {
            indexList.add(-1);
        }
        this.N = n;
    }

    public void solve(ArrayList<City> cityList) {
        int n = this.N;
        // 城市空间分布信息
        cities = new double[n][2];
        for (int i=0; i<n; i++) {
            cities[i][0] = cityList.get(i).getX();
            cities[i][1] = cityList.get(i).getY();
        }

        // TSP距离矩阵
        graph = new double[n][n];
        for (int i=0; i<n; i++) {
            for (int j=i; j<n; j++) {
                if (i==j) {
                    graph[i][j] = 0.0;
                    continue;
                }
                double x1 = cityList.get(i).getX();
                double y1 = cityList.get(i).getY();
                double x2 = cityList.get(j).getX();
                double y2 = cityList.get(j).getY();
                graph[i][j] = Math.sqrt(Math.pow((x1-x2),2)+Math.pow((y1-y2),2));
                graph[j][i] = graph[i][j];
                //System.out.println(graph[i][j]);
            }
        }

        // 使用DP求解
        solver = new TSPSolver();
        this.optIndexList = solver.computeTSP(graph);
        this.optDist = solver.shortest;
    }

    public void setAllActions(ArrayList<Integer> list) {
        if ( N != list.size()-1 || list.get(0)!=0 || list.get(list.size()-1)!=0) {
            System.out.println("动作序列设置有误");
            return;
        }
        this.indexList = list;
    }

    public void setAction(int action) {
        if (index > this.N)
            return;
        this.indexList.set(index, action);
    }

    public int getIndex() {
        if (index < indexList.size()) {
            return indexList.get(index);
        }
        else {
            // 用-2标识回到了原点
            return -2;
        }
    }

    public void setNextIndex() {
        this.index++;
    }

    public void addDistance(double dist) {
        this.distance += dist;
    }

    public double getDistance() {
        return this.distance;
    }

    public double getOptDist() {
        return this.optDist;
    }

    public ArrayList<Integer> getIndexList() {
        return indexList;
    }

    public String getIndexListString() {
        return this.indexList.toString();
    }

    public String getOptIndexListString() {
        return this.optIndexList.toString();
    }

    public double[][] getCities() {
        return this.cities;
    }

    public double getReward() {
        return this.distance - this.optDist;
    }
}