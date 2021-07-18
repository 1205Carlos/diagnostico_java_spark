package minsait.ttaa.datio.engine;

import static minsait.ttaa.datio.common.Common.HEADER;
import static minsait.ttaa.datio.common.Common.INFER_SCHEMA;
import static minsait.ttaa.datio.common.Common.INPUT_PATH;
import static minsait.ttaa.datio.common.Common.PARAMS_PATH;
import static minsait.ttaa.datio.common.Common.PARAM_WORK_DS;
import static minsait.ttaa.datio.common.naming.PlayerCategory.CAT_A;
import static minsait.ttaa.datio.common.naming.PlayerCategory.CAT_B;
import static minsait.ttaa.datio.common.naming.PlayerCategory.CAT_C;
import static minsait.ttaa.datio.common.naming.PlayerCategory.CAT_D;
import static minsait.ttaa.datio.common.naming.PlayerInput.age;
import static minsait.ttaa.datio.common.naming.PlayerInput.clubName;
import static minsait.ttaa.datio.common.naming.PlayerInput.heightCm;
import static minsait.ttaa.datio.common.naming.PlayerInput.longName;
import static minsait.ttaa.datio.common.naming.PlayerInput.nationality;
import static minsait.ttaa.datio.common.naming.PlayerInput.overall;
import static minsait.ttaa.datio.common.naming.PlayerInput.potential;
import static minsait.ttaa.datio.common.naming.PlayerInput.shortName;
import static minsait.ttaa.datio.common.naming.PlayerInput.teamPosition;
import static minsait.ttaa.datio.common.naming.PlayerInput.weight_kg;
import static minsait.ttaa.datio.common.naming.PlayerOutput.catHeightByPosition;
import static minsait.ttaa.datio.common.naming.PlayerOutput.playerCat;
import static minsait.ttaa.datio.common.naming.PlayerOutput.potentialVsOverall;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.rank;
import static org.apache.spark.sql.functions.when;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.TestOnly;

public class Transformer extends Writer {
		private final static double MIN_POT_OVER_C = 1.15;
		private final static double MIN_POT_OVER_D = 1.25;
	
    private SparkSession spark;

    public Transformer(@NotNull SparkSession spark) {
        this.spark = spark;
        Dataset<Row> df = readInput();

        df.printSchema();

        df = cleanData(df);
        df = columnSelection(df);
        
        String param23Years = getParameterFile(PARAM_WORK_DS);
        
        //Ejercicio 5
        df = apply23YearsFilterDataset(df, Integer.parseInt(param23Years));

        //Ejercicio 2
        df = playerCatFunction(df);
        
        //Ejercicio 3
        df = potentialVsOverallFunction(df);
        
        //Ejercicio 4
        df = filterByCatAndPotentialOverallFunction(df);
        
        df.show(100,false);
        
        
        //Test ejercicio 4
        try {
        	testFilterByCatAndPotentialOverallFunction(df);
        }catch(Exception e) {
        	return;
        }
        
        // Uncomment when you want write your final output
        //Ejercicio 6
        write(df);
    }

    private Dataset<Row> columnSelection(Dataset<Row> df) {
    	//Ejercicio 1
        return df.select(
                shortName.column(),
                longName.column(),
                age.column(),
                heightCm.column(),
                weight_kg.column(),
                nationality.column(),
                clubName.column(),
                overall.column(),
                potential.column(),
                teamPosition.column()
        );
    }

    /**
     * @return a Dataset readed from csv file
     */
    private Dataset<Row> readInput() {
        Dataset<Row> df = spark.read()
                .option(HEADER, true)
                .option(INFER_SCHEMA, true)
                .csv(INPUT_PATH);
        return df;
    }
    
    private String getParameterFile(String paramName) {
    	try(Stream<String> stream = Files.lines(Paths.get(PARAMS_PATH))) {
    		String value = stream
    				.filter(line -> line.startsWith(paramName))
    				.collect(Collectors.toList())
    				.get(0);
    		
    		value = value.substring(paramName.length() + 1);
    		return value;
    	}catch(IOException e) {
    		e.printStackTrace();
    		return null;
    	}
    }
    
    private Dataset<Row> apply23YearsFilterDataset(Dataset<Row> df, int param) {
    	if(param == 1) {
    		return df.filter(age.column().lt(23));
    	}
    	
    	return df;
    }

    /**
     * @param df
     * @return a Dataset with filter transformation applied
     * column team_position != null && column short_name != null && column overall != null
     */
    private Dataset<Row> cleanData(Dataset<Row> df) {
        df = df.filter(
                teamPosition.column().isNotNull().and(
                        shortName.column().isNotNull()
                ).and(
                        overall.column().isNotNull()
                )
        );

        return df;
    }

    /**
     * @param df is a Dataset with players information (must have team_position and height_cm columns)
     * @return add to the Dataset the column "cat_height_by_position"
     * by each position value
     * cat A for if is in 20 players tallest
     * cat B for if is in 50 players tallest
     * cat C for the rest
     */
    private Dataset<Row> exampleWindowFunction(Dataset<Row> df) {
        WindowSpec w = Window
                .partitionBy(teamPosition.column())
                .orderBy(heightCm.column().desc());

        Column rank = rank().over(w);

        Column rule = when(rank.$less(10), "A")
                .when(rank.$less(50), "B")
                .otherwise("C");

        df = df.withColumn(catHeightByPosition.getName(), rule);

        return df;
    }
    
    private Dataset<Row> playerCatFunction(Dataset<Row> df){
    	WindowSpec w = Window
    					.partitionBy(nationality.column(), teamPosition.column())
    					.orderBy(overall.column().desc());
    	
    	Column rank = rank().over(w);
    	Column rule = when(rank.$less(3), CAT_A)
    							.when(rank.$less(5), CAT_B)
    							.when(rank.$less(10), CAT_C)
    							.otherwise(CAT_D);
    	
    	df = df.withColumn(playerCat.getName(), rule);
    	return df;
    }
    
    private Dataset<Row> potentialVsOverallFunction(Dataset<Row> df){
    	Column rule = col(potential.getName())
    									.divide(col(overall.getName()));
    	
    	df = df.withColumn(potentialVsOverall.getName(), rule);
    	return df;
    }
    
    private Dataset<Row> filterByCatAndPotentialOverallFunction(Dataset<Row> df){
    	Column ruleA = col(playerCat.getName())
    			.equalTo("A");
    	
    	Column ruleB = col(playerCat.getName())
    			.equalTo("B");
    	
    	//Rules for A and B
    	Column ruleAB = ruleA.or(ruleB);
    	
    	Column ruleC = col(playerCat.getName()).equalTo("C")
    			.and(col(potentialVsOverall.getName())
    			.gt(MIN_POT_OVER_C));
    	
    	Column ruleD = col(playerCat.getName()).equalTo("D")
    			.and(col(potentialVsOverall.getName())
    			.gt(MIN_POT_OVER_D));	
    	
    	//Rules for C and D
    	Column ruleCD = ruleC.or(ruleD);
    	
    	//Join all rules
    	Column rules = ruleAB.or(ruleCD);
    	
    	df = df.filter(rules);
    	return df;
    }
    
    @TestOnly
    void testFilterByCatAndPotentialOverallFunction(Dataset<Row> df) throws Exception{
    	
    	df.foreach((ForeachFunction<Row>) s -> {
    		String playerCatRow = s.getAs(playerCat.getName());
    		double potOver = s.getAs(potentialVsOverall.getName());
    		
    		if(playerCatRow.equals(CAT_C)) {
    			if(potOver < MIN_POT_OVER_C) {
    				throw new Exception();
    			}
    		}else if(playerCatRow.equals(CAT_D)) {
    			if(potOver < MIN_POT_OVER_D) {
    				throw new Exception();
    			}
    		}
    	});
    }
}
