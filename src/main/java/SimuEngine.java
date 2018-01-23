import java.util.ArrayList;

public class SimuEngine {
    // 仿真钟
    private SimuTimer simuTimer = null;
    // 事件表
    private ArrayList<SimuEvent> eventList = null;
    // 仿真实体
    ArrayList<City> cityList;
    private Rider rider;
    // 实验模块
    private Experiment exp = null;

    public SimuEngine() {
    }

    public void initialize(Experiment exp) {
        simuTimer = new SimuTimer();

        // 假设TSP有n个地址
        int n = exp.N;
        if (n==0) {
            System.out.println("未设置实验数量");
            return;
        }
        cityList = new ArrayList<>();
        for (int i=0; i<n; i++) {
            cityList.add(new City(i));
            // 输出商店位置坐标
            //System.out.println(Double.toString(cityList.get(i).getX())+","+Double.toString(cityList.get(i).getY()));
        }

        this.exp = exp;
        exp.solve(cityList);

        cityList.add(cityList.get(0));    // 回到原点

        // 1个骑手并初始化位置
        rider = new Rider(0, cityList.get(exp.getIndex()).getX(), cityList.get(exp.getIndex()).getY());
        //System.out.println("Arrival: 0");
        exp.setNextIndex();

        // 初始化事件队列，插入1个MOVE事件
        eventList = new ArrayList<>();
        eventList.add(new SimuEvent(EventType.MOVE));
    }

    public int run(int action) {
        if (this.simuTimer == null) {
            System.out.println("错误：仿真引擎未初始化");
            return -1;
        }

        // 若action=-1，则一次全部执行
        if (action >= 0){
            exp.setAction(action);
        }

        while (exp.getIndex() >= 0) {
            // 仿真时间运转
            simuTimer.timing(eventList);
            // 事件处理
            switch (simuTimer.getNextEventType()) {
                case IDLE:
                    break;
                case ARRIVAL:
                    arrivalEventHandler();
                    break;
                case MOVE:
                    moveEvnetHandler(rider, cityList.get(exp.getIndex()).getX(), cityList.get(exp.getIndex()).getY());
                    break;
            }
        }

        if (true) {
            // 输出统计结果
            System.out.println("Simu path:" + exp.getIndexListString());
            System.out.println(exp.getDistance());
            // 输出理论计算结果
            System.out.println("Opti path:" + exp.getOptIndexListString());
            System.out.println(exp.getOptDist());
        }

        // 判断仿真结束
        if (exp.getIndex() == -2) {
            return 1;
        }
        // 仿真中止
        return 0;
    }

    // 仿真模型逻辑
    private void arrivalEventHandler() {
        //System.out.println("Arrival: "+Integer.toString(exp.getIndex()));
        eventList.remove(0);
        exp.setNextIndex();
    }

    private void moveEvnetHandler(Rider rider, double x, double y) {
        double dist = rider.move(x, y);
        exp.addDistance(dist);
        //System.out.println("Move: "+rider.getPosition());
        eventList.remove(0);

        if (rider.getDistance(x, y) < rider.getSpeed()) {
            eventList.add(0, new SimuEvent(EventType.MOVE));
            eventList.add(0, new SimuEvent(EventType.ARRIVAL));
        } else {
            eventList.add(0, new SimuEvent(EventType.MOVE));
        }
    }
}