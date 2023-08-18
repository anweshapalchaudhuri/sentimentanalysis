package com.anwesha.dissertation;

import org.apache.commons.collections4.functors.InstanceofPredicate;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PiePlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DefaultPieDataset;
import org.json.JSONObject;
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
import javafx.application.Application;

import java.awt.Color;
import java.awt.Font;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import javax.swing.JFrame;

	public class JsoupWebScrap{
	    public static void main(String[] args) {
	    	String timesharePropertyName = "Westgate Las Vegas Resort & Casino";
	    	List<String> reviewTextList = new ArrayList();
	    	String jsonString = "";
	    	ObjectMapper objectMapper = new ObjectMapper();
	    	
	        try {
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
//	        	   JsonNode jnDataContent = jnDataContent1.get("locations");
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
	           //Map<String, Integer> insights = analyzeSentiments(reviewTextList, stanPipeLne, timesharePropertyName);
//	           analyzeAspectsAndSentiments(reviewTextList,stanPipeLne);
	           List<Map<String, Double>> aspectSentimentScores = calculateAspectSentimentScores(reviewTextList, stanPipeLne);

	           // Get the list of unique aspect names
	           Set<String> aspectSet = new HashSet<>();
	           for (Map<String, Double> aspectSentimentScore : aspectSentimentScores) {
	               aspectSet.addAll(aspectSentimentScore.keySet());
	           }
	           List<String> aspectNames = new ArrayList<>(aspectSet);

	           double[][] aspectSentimentMatrix = createAspectSentimentMatrix(aspectSentimentScores, aspectNames);
	           double[][] correlationMatrix = calculateCorrelationMatrix(aspectSentimentMatrix);

	           System.out.println("Aspect names:");
	           for (int i = 0; i < aspectNames.size(); i++) {
	               System.out.printf("%d: %s%n", i, aspectNames.get(i));
	           }

	           System.out.println("\nCorrelation matrix:");
	           System.out.print("       ");
	           for (int i = 0; i < aspectNames.size(); i++) {
	               System.out.printf("A%-4d ", i);
	           }
	           System.out.println();

	           for (int i = 0; i < correlationMatrix.length; i++) {
	               System.out.printf("A%-4d ", i);
	               for (int j = 0; j < correlationMatrix[i].length; j++) {
	                   System.out.printf("%.2f ", correlationMatrix[i][j]);
	               }
	               System.out.println();
	           }
	       
	           

	        } catch (IOException e) {
	            e.printStackTrace();
	        }
	    }
	    
	    
	    private static StanfordCoreNLP createCoreNlpPipeline() {
	        Properties props = new Properties();
	        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, sentiment");
	        return new StanfordCoreNLP(props);
	    }

	    public static Map<String, Integer> analyzeSentiments(List<String> reviews, StanfordCoreNLP pipeline, String header) {
	        Map<String, Integer> sentimentCounts = new HashMap<>();

	        for (String review : reviews) {
	            CoreDocument document = new CoreDocument(review);
	            pipeline.annotate(document);

	            for (CoreSentence sentence : document.sentences()) {
	                String sentiment = sentence.sentiment();
	                sentimentCounts.put(sentiment, sentimentCounts.getOrDefault(sentiment, 0) + 1);
	            }
	        }

	        // Generate a chart using the sentiment counts, for example, a pie chart for sentiment distribution
	        generatePieChart(sentimentCounts, header);
	        return sentimentCounts;
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
	    
	    public static void generateBarChart(Map<String, Map<String, Integer>> aspectSentimentCounts) {
	        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

	        for (Map.Entry<String, Map<String, Integer>> aspectEntry : aspectSentimentCounts.entrySet()) {
	            String aspect = aspectEntry.getKey();
	            Map<String, Integer> sentimentCounts = aspectEntry.getValue();

	            for (Map.Entry<String, Integer> sentimentEntry : sentimentCounts.entrySet()) {
	            	System.out.println("sentimentEntry.getValue() "+ sentimentEntry.getValue() + " sentimentEntry.getKey() "+ sentimentEntry.getKey()+ " aspect --> "+aspect);
	                dataset.addValue(sentimentEntry.getValue(), sentimentEntry.getKey(), aspect);
	            }
	        }

	        JFreeChart barChart = ChartFactory.createBarChart(
	            "Aspect-Based Sentiment Distribution",
	            "Aspects",
	            "Count",
	            dataset
	        );
	        
	      //Increase the font size of the category axis labels
//	        CategoryPlot plot = (CategoryPlot) barChart.getPlot();
//	        CategoryAxis categoryAxis = plot.getDomainAxis();
//	        Font newFont = new Font("SansSerif", Font.BOLD, 6);
//	        categoryAxis.setTickLabelFont(newFont);

	        
	        CategoryPlot plot = (CategoryPlot) barChart.getPlot();
	        BarRenderer renderer = (BarRenderer) plot.getRenderer();
	        renderer.setSeriesPaint(dataset.getRowIndex("Negative"), Color.RED); 
	        renderer.setSeriesPaint(dataset.getRowIndex("Positive"), Color.GREEN);
	        renderer.setSeriesPaint(dataset.getRowIndex("Very positive"), Color.BLUE);
	        renderer.setSeriesPaint(dataset.getRowIndex("Neutral"), Color.YELLOW);
	        
	        ChartPanel chartPanel = new ChartPanel(barChart);
	        chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
	        JFrame frame = new JFrame();
	        frame.add(chartPanel);
	        frame.pack();
	        frame.setVisible(true);
	    }
	    
	    public static void generatePieChart(Map<String, Integer> sentimentCounts, String header) {
	        DefaultPieDataset dataset = new DefaultPieDataset();

	        for (Map.Entry<String, Integer> entry : sentimentCounts.entrySet()) {
	            dataset.setValue(entry.getKey(), entry.getValue());
	        }

	        JFreeChart pieChart = ChartFactory.createPieChart(
	            header,
	            dataset,
	            true,
	            true,
	            false
	        );

	        PiePlot plot = (PiePlot) pieChart.getPlot();
	        ChartPanel chartPanel = new ChartPanel(pieChart);
	        chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
	        JFrame frame = new JFrame();
	        frame.add(chartPanel);
	        frame.pack();
	        frame.setVisible(true);
	    }
	    
	    
	    public static double[][] calculateCorrelationMatrix(double[][] aspectSentimentMatrix) {
	    	PearsonsCorrelation correlation = new PearsonsCorrelation(aspectSentimentMatrix);
	        return correlation.getCorrelationMatrix().getData();
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
	    
	    
	    public static double[][] createAspectSentimentMatrix(List<Map<String, Double>> aspectSentimentScores, List<String> aspectNames) {
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

	    
	    
	    public static List<Map<String, Double>> calculateAspectSentimentScores(List<String> reviews, StanfordCoreNLP pipeline ) {
	        List<Map<String, Double>> aspectSentimentScores = new ArrayList<>();

	        for (String review : reviews) {
	            Map<String, Map<String, Integer>> aspectSentimentCounts = analyzeAspectsAndSentiments(Arrays.asList(review), pipeline);
	            Map<String, Double> aspectSentimentAverages = calculateAspectSentimentAverages(aspectSentimentCounts);
	            aspectSentimentScores.add(aspectSentimentAverages);
	        }

	        return aspectSentimentScores;
	    }
	    
	    
	    
	}

	

