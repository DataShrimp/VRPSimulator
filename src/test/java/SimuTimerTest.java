import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class SimuTimerTest {
    @Test
    public void timing() throws Exception {
        SimuTimer timer = new SimuTimer();
        ArrayList<SimuEvent> events= new ArrayList<>();
        events.add(new SimuEvent(EventType.ARRIVAL, 0));
        events.add(new SimuEvent(EventType.MOVE, 1));
        timer.timing(events);
        System.out.println(timer.getSimuTime());
    }

}