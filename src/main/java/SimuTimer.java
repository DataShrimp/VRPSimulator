import java.util.ArrayList;

public class SimuTimer {
    private double simuTime = 0.0;
    private EventType nextEventType = null;

    public void timing(ArrayList<SimuEvent> nextEvents) {
        double minNextEventTime = 1e29;

        if (nextEvents.isEmpty()) {
            nextEventType = EventType.IDLE;
            return;
        }

        for (SimuEvent event: nextEvents) {
            if (event.getEventTime() < minNextEventTime) {
                minNextEventTime = event.getEventTime();
                nextEventType = event.getEventType();
            }
        }

        this.simuTime = minNextEventTime;
    }

    public double getSimuTime() {
        return this.simuTime;
    }

    public EventType getNextEventType() {
        return this.nextEventType;
    }

}