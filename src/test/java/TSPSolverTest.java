import org.junit.Test;

public class TSPSolverTest {

    @Test
    public void computeTSP() {
        TSPSolver solver = new TSPSolver();
        double[][] test = {{0,3,6,7},{5,0,2,3},{6,4,0,2},{3,7,5,0}};
        solver.computeTSP(test);
        System.out.println(solver.shortest);
    }
}