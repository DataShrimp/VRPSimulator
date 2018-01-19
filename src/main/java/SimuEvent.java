public class SimuEvent {
    private double time = 0.0;
    private EventType type = null;

    public SimuEvent(EventType type){
        this.type = type;
    }

    public SimuEvent(EventType type, double time) {
        this.type = type;
        this.time = time;
    }

    public EventType getEventType() {
        return this.type;
    }

    public void setEventTime(double time) {
        this.time = time;
    }

    public double getEventTime() {
        return this.time;
    }
}
