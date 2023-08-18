package com.anwesha.dissertation;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.text.Font;
import javafx.stage.Stage;

import javax.imageio.ImageIO;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.jsoup.Jsoup;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreEntityMention;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class CorrelationMatrixImage extends Application {
    // ...
    
	public static void main(String[] args) {
		Application.launch(args);
	}
	
    @Override
    public void start(Stage primaryStage) throws IOException {
        // Assuming you have a list of reviews
//        List<String> reviews = Arrays.asList(
//            "I loved this product!",
//            "The service was terrible, I'll never come back.",
//            "The food was mediocre, but the ambiance was great."
//        );
    	
    	
    	String timesharePropertyName = "Westgate Las Vegas Resort & Casino";
    	List<String> reviewTextList = new ArrayList();
    	String jsonString = "";
    	ObjectMapper objectMapper = new ObjectMapper();
    	
        //try {
        	// https://www.tripadvisor.com/Hotels-g191-United_States-Hotels.html
        	
            // String url = "https://www.reddit.com/r/java/top/?t=day";
        	String url = "https://www.tripadvisor.com/Hotel_Review-g45963-d91878-Reviews-Westgate_Las_Vegas_Resort_Casino-Las_Vegas_Nevada.html";
            Document doc = Jsoup.connect(url)
            		.ignoreContentType(true)
            		.referrer("https://www.google.com")
            		.followRedirects(true)
            		.userAgent("WhatsApp/2.19.81 A")
            		.timeout(30000).get();
            //Elements posts = doc.select("div.Post");
            //Elements posts = doc.getElementsByTag("div").select("div[data-hotels-data]");
            Elements scriptElements = doc.getElementsByTag("script");
            for (Element element :scriptElements ){
            for (DataNode node : element.dataNodes()) {
                jsonString = node.getWholeData();
            }
            }
            
            // Replace "window.__WEB_CONTEXT__=" with "" in jsonString
            
            String newText = jsonString.replaceFirst("window.__WEB_CONTEXT__=", "");
            
            String patternTrimText = Pattern.quote("(this.$WP=this.$WP||[]).push(['@ta/features',function(e){return [function(){e('default',__WEB_CONTEXT__.pageManifest.features);},[]]},[]]);");
            
            //Also replace (this.$WP=this.$WP||[]).push(['@ta/features',function(e){return [function(){e('default',__WEB_CONTEXT__.pageManifest.features);},[]]},[]]); with ""

            
            String jsonTextString = newText.replaceFirst(patternTrimText, "");
            
            String finalString = jsonTextString.replaceFirst("pageManifest", " \"pageManifest\" ");
            
            JsonNode jsonNode = objectMapper.readTree(finalString);
            JsonNode pageManifest = jsonNode.get("pageManifest");
            JsonNode urqlCache = pageManifest.get("urqlCache");
            JsonNode results = urqlCache.get("results");
            
            List<JsonNode> datasets = StreamSupport
            	    .stream(results.spliterator(), false)
            	    .collect(Collectors.toList());
            
           for(JsonNode jsn: datasets) {
        	   
        	   // System.out.println(jsn.toPrettyString()+"yyyxxxxxxyyy");
        	   JsonNode jn = jsn.get("data");
        	   JsonNode jnDataContent1 = objectMapper.readTree(jn.textValue());
        	   JsonNode jnDataContent = jnDataContent1.get("locations");
//        	   JsonNode jnDataContent = jnDataContent1.get("locations");
        	   if(null != jnDataContent) {
        		   List<JsonNode> reviews = StreamSupport
   	            	    .stream(jnDataContent.spliterator(), false)
   	            	    .collect(Collectors.toList());
        		   for(JsonNode review: reviews) {
        			   JsonNode jnReviewData = review.get("reviewListPage");
        			   if(null != jnReviewData) {
        				   JsonNode reviewsDataSet = jnReviewData.get("reviews");
        				   
        				   if(null != reviewsDataSet) {
        					   List<JsonNode> reviewList = StreamSupport
        		   	            	    .stream(reviewsDataSet.spliterator(), false)
        		   	            	    .collect(Collectors.toList());
        					   
        					   for(JsonNode reviewData:reviewList) {
        						   JsonNode reviewText = reviewData.get("text");
        						   reviewTextList.add(reviewText.asText());
        						   
        					   }
        				   }

        				   
        			   }
        			   
        		   }
        		   
        	   }
        	   
           }

    	
           StanfordCoreNLP stanPipeLne = createCoreNlpPipeline();
    	
    	

        List<Map<String, Double>> aspectSentimentScores = calculateAspectSentimentScores(reviewTextList, stanPipeLne);
        Set<String> aspectSet = new HashSet<>();
        for (Map<String, Double> aspectSentimentScore : aspectSentimentScores) {
            aspectSet.addAll(aspectSentimentScore.keySet());
        }
        List<String> aspectNames = new ArrayList<>(aspectSet);
        double[][] aspectSentimentMatrix = createAspectSentimentMatrix(aspectSentimentScores, aspectNames);
        double[][] correlationMatrix = calculateCorrelationMatrix(aspectSentimentMatrix);

        GridPane gridPane = createCorrelationMatrixGridPane(correlationMatrix, aspectNames);
        Scene scene = new Scene(gridPane);
        WritableImage image = scene.snapshot(null);

        ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", new File("correlation_matrix.png"));

        primaryStage.close();
    }
    
        public static List<Map<String, Double>> calculateAspectSentimentScores(List<String> reviews, StanfordCoreNLP pipeline ) {
	        List<Map<String, Double>> aspectSentimentScores = new ArrayList<>();

	        for (String review : reviews) {
	            Map<String, Map<String, Integer>> aspectSentimentCounts = analyzeAspectsAndSentiments(Arrays.asList(review), pipeline);
	            Map<String, Double> aspectSentimentAverages = calculateAspectSentimentAverages(aspectSentimentCounts);
	            aspectSentimentScores.add(aspectSentimentAverages);
	        }

	        return aspectSentimentScores;
	    }
        
        
        public static Map<String, Double> calculateAspectSentimentAverages(Map<String, Map<String, Integer>> aspectSentimentCounts) {
	        Map<String, Double> aspectSentimentAverages = new HashMap<>();

	        for (Map.Entry<String, Map<String, Integer>> aspectEntry : aspectSentimentCounts.entrySet()) {
	            String aspect = aspectEntry.getKey();
	            Map<String, Integer> sentimentCounts = aspectEntry.getValue();

	            double totalSentiment = 0.0;
	            int totalReviews = 0;

	            for (Map.Entry<String, Integer> sentimentEntry : sentimentCounts.entrySet()) {
	                String sentiment = sentimentEntry.getKey();
	                int count = sentimentEntry.getValue();

	                // Assign numerical values to each sentiment category (e.g., -1 for negative, 0 for neutral, 1 for positive)
	                int sentimentValue = 0;
	                if (sentiment.equalsIgnoreCase("positive")) {
	                    sentimentValue = 1;
	                } else if (sentiment.equalsIgnoreCase("negative")) {
	                    sentimentValue = -1;
	                }

	                totalSentiment += sentimentValue * count;
	                totalReviews += count;
	            }

	            double averageSentiment = totalSentiment / totalReviews;
	            aspectSentimentAverages.put(aspect, averageSentiment);
	        }

	        return aspectSentimentAverages;
	    }
        
        
        private static StanfordCoreNLP createCoreNlpPipeline() {
	        Properties props = new Properties();
	        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, sentiment");
	        return new StanfordCoreNLP(props);
	    }
        
        
        public static Map<String, Map<String, Integer>> analyzeAspectsAndSentiments(List<String> reviews, StanfordCoreNLP pipeline) {
	        Map<String, Map<String, Integer>> aspectSentimentCounts = new HashMap<>();

	        for (String review : reviews) {
	            CoreDocument document = new CoreDocument(review);
	            pipeline.annotate(document);

	            for (CoreSentence sentence : document.sentences()) {
	                List<String> aspects = extractAspects(sentence);
	                String sentiment = sentence.sentiment();

	                for (String aspect : aspects) {
	                    Map<String, Integer> sentimentCounts = aspectSentimentCounts.computeIfAbsent(aspect, k -> new HashMap<>());
	                    sentimentCounts.put(sentiment, sentimentCounts.getOrDefault(sentiment, 0) + 1);
	                }
	            }
	        }
	    
	     // Generate a chart using the aspect sentiment counts, for example, a stacked bar chart for sentiment distribution per aspect
	        //generateBarChart(aspectSentimentCounts);
	        return aspectSentimentCounts;
	    }
        
        
        private static List<String> extractAspects(CoreSentence sentence) {
	        List<String> aspects = new ArrayList<>();

	        for (CoreEntityMention entity : sentence.entityMentions()) {
	            // You may need to filter entities based on entityType or other criteria depending on your use case
	            aspects.add(entity.text());
	        }

	        return aspects;
	    }
        
    public static double[][] calculateCorrelationMatrix(double[][] aspectSentimentMatrix) {
    	PearsonsCorrelation correlation = new PearsonsCorrelation(aspectSentimentMatrix);
        return correlation.getCorrelationMatrix().getData();
    }
    
    
    public static double[][] createAspectSentimentMatrix(List<Map<String, Double>> aspectSentimentScores, List<String> aspectNames) {
    	
    	aspectSentimentScores.forEach(item -> item.forEach((k, v) -> System.out.println(k + ": " + v)));
    	aspectNames.forEach(item -> System.out.println(item));
        int numAspects = aspectNames.size();
        int numReviews = aspectSentimentScores.size();
        double[][] matrix = new double[numReviews][numAspects];

        for (int i = 0; i < numReviews; i++) {
            Map<String, Double> aspectSentimentScore = aspectSentimentScores.get(i);
            for (int j = 0; j < numAspects; j++) {
                String aspect = aspectNames.get(j);
                matrix[i][j] = aspectSentimentScore.getOrDefault(aspect, 0.0);
            }
        }

        return matrix;
    }
    
    
    
    private GridPane createCorrelationMatrixGridPane(double[][] correlationMatrix, List<String> aspectNames) {
        GridPane gridPane = new GridPane();
        gridPane.setHgap(5);
        gridPane.setVgap(5);
        gridPane.setPadding(new Insets(10));

        for (int i = 0; i < correlationMatrix.length; i++) {
            for (int j = 0; j < correlationMatrix[i].length; j++) {
                Label cellLabel = new Label(String.format("%.2f", correlationMatrix[i][j]));
                cellLabel.setFont(Font.font("Arial", 14));
                cellLabel.setStyle("-fx-background-color: white; -fx-border-color: black; -fx-padding: 5;");
                gridPane.add(cellLabel, j + 1, i + 1);
            }
        }

        for (int i = 0; i < aspectNames.size(); i++) {
            Label rowLabel = new Label("A" + i);
            rowLabel.setFont(Font.font("Arial", 14));
            rowLabel.setStyle("-fx-background-color: white; -fx-border-color: black; -fx-padding: 5;");
            gridPane.add(rowLabel, 0, i + 1);

            Label colLabel = new Label("A" + i);
            colLabel.setFont(Font.font("Arial", 14));
            colLabel.setStyle("-fx-background-color: white; -fx-border-color: black; -fx-padding: 5;");
            gridPane.add(colLabel, i + 1, 0);
        }

        return gridPane;
    }

    
}
