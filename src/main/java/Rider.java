public class Rider {
    private int id = 0;
    private double x,y;
    // 每秒距离，距离最大范围[0,1]
    private double speed = 0.001;

    public Rider(int id, double x, double y) {
        this.id = id;
        this.x = x;
        this.y = y;
    }

    public double move(double xx, double yy) {
        // 计算方向
        double dist = getDistance(xx, yy);
        double dirX = (xx-this.x)/dist;
        double dirY = (yy-this.y)/dist;
        // 更新距离
        this.x += dirX*speed;
        this.y += dirY*speed;

        return speed;
    }

    public double getDistance(double xx, double yy) {
        double dist = Math.sqrt(Math.pow((xx-this.x),2)+Math.pow((yy-this.y),2));
        return dist;
    }

    public double getSpeed() {
        return this.speed;
    }

    public String getPosition() {
        return Double.toString(x)+","+Double.toString(y);
    }
}
