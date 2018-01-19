import org.junit.Test;

import static org.junit.Assert.*;

public class RandomGeneratorTest {
    @Test
    public void getRandom() throws Exception {
        for (int i=0; i<10; i++)
            System.out.println(new RandomGenerator().getRandom());
    }

}