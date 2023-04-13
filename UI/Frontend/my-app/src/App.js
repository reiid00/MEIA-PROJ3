import React, { useState } from "react";
import { VStack, Box, Heading, TextArea, Button, Text, Modal } from "native-base";

const primaryColor ="orange.600";

const App = () => {
  const [complaint, setComplaint] = useState("");
  const [response, setResponse] = useState(null);

  const handleSubmit = async () => {
    // Make API request to submit ticket with complaint
    // const response = await fetch("http://localhost:8081/submitTicket", {
    //   method: "POST",
    //   body: JSON.stringify({ complaint }),
    //   headers: {
    //     "Content-Type": "application/json",
    //   },
    // });

    // Parse response as JSON
    // const data = await response.json();
    const data = {
      Answer:"Nice",
      PredictedEmotionalSentiment:"angry",
      PredictedCategories:"Stonks, Sub-stonks",
      DetectedLanguage:"pt-en",
    }
    setResponse(data);
  };

  return (
    <VStack flex={1} justifyContent="center" alignItems="center" p={4}>
      <Heading mb={4} color={primaryColor}>Support System</Heading>
      <Box w="100%">
        <TextArea
          h={200}
          color={"black"}
          bordered
          placeholder="Write your complaint here"
          value={complaint}
          onChangeText={setComplaint}
        />
        <Button bgColor={primaryColor} mt={4} onPress={handleSubmit} isDisabled={!complaint}>
          Submit
        </Button>
      </Box>

      {response && (
        <Modal isOpen={response} onClose={() => setResponse(null)} bgColor={primaryColor} closeOnOverlayClick={false}>
          <Box p={4}>
            <Heading size="md">Ticket Response</Heading>
            <Box mt={2}>
              <Text>Answer: {response.Answer}</Text>
              <Text>Predicted Emotional Sentiment: {response.PredictedEmotionalSentiment}</Text>
              <Text>Predicted Categories: {response.PredictedCategories}</Text>
              <Text>Detected Language: {response.DetectedLanguage}</Text>
            </Box>
            <Button mt={4} onPress={() => setResponse(null)}>
              Close
            </Button>
          </Box>
        </Modal>
      )}
    </VStack>
  );
};

export default App;