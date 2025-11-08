package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
)

type GameState struct {
	Board         [][]int      `json:"board"`
	Agent1Trail   [][]int      `json:"agent1_trail"`
	Agent2Trail   [][]int      `json:"agent2_trail"`
	Agent1Length  int          `json:"agent1_length"`
	Agent2Length  int          `json:"agent2_length"`
	Agent1Alive   bool         `json:"agent1_alive"`
	Agent2Alive   bool         `json:"agent2_alive"`
	Agent1Boosts  int          `json:"agent1_boosts"`
	Agent2Boosts  int          `json:"agent2_boosts"`
	TurnCount     int          `json:"turn_count"`
	PlayerNumber  int          `json:"player_number"`
}

var globalGameState GameState

func infoHandler(w http.ResponseWriter, r *http.Request) {
	participant := getEnv("PARTICIPANT", "SteamrollerParticipant")
	agentName := getEnv("AGENT_NAME", "SteamrollerAgentV0")
	
	response := map[string]string{
		"participant": participant,
		"agent_name":  agentName,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func receiveStateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var state GameState
	if err := json.NewDecoder(r.Body).Decode(&state); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	globalGameState = state
	
	response := map[string]string{"status": "state received"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func sendMoveHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	playerNumberStr := r.URL.Query().Get("player_number")
	playerNumber := 1
	if playerNumberStr != "" {
		if pn, err := strconv.Atoi(playerNumberStr); err == nil {
			playerNumber = pn
		}
	}
	
	var myTrail [][]int
	var myBoosts int
	var otherTrail [][]int
	
	if playerNumber == 1 {
		myTrail = globalGameState.Agent1Trail
		myBoosts = globalGameState.Agent1Boosts
		otherTrail = globalGameState.Agent2Trail
	} else {
		myTrail = globalGameState.Agent2Trail
		myBoosts = globalGameState.Agent2Boosts
		otherTrail = globalGameState.Agent1Trail
	}
	
	move := DecideMove(myTrail, otherTrail, globalGameState.TurnCount, myBoosts)
	
	response := map[string]string{"move": move}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func endGameHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var endData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&endData); err == nil {
		if result, ok := endData["result"].(string); ok {
			fmt.Printf("\nGame Over! Result: %s\n", result)
		}
	}
	
	response := map[string]string{"status": "acknowledged"}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func main() {
	http.HandleFunc("/", infoHandler)
	http.HandleFunc("/send-state", receiveStateHandler)
	http.HandleFunc("/send-move", sendMoveHandler)
	http.HandleFunc("/end", endGameHandler)
	
	port := getEnv("PORT", "5008")
	
	agentName := getEnv("AGENT_NAME", "SteamrollerAgent")
	participant := getEnv("PARTICIPANT", "SteamrollerParticipant")
	
	fmt.Printf("Starting %s (%s) on port %s...\n", agentName, participant, port)
	
	addr := "0.0.0.0:" + port
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal(err)
	}
}
