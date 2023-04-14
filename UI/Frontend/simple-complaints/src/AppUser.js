import React, { useState, useContext } from "react";
import {
  Container,
  Box,
  Typography,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from "@mui/material";
import axios from "axios";
import { TicketContext } from "./TicketContext";
import { useNavigate } from "react-router-dom";
import './App.css';

function AppUser() {
  const [complaintText, setComplaintText] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [responseInfo, setResponseInfo] = useState(null);
  const [tickets, setTickets] = useContext(TicketContext);
  const navigate = useNavigate();

  const handleSubmit = async () => {
    try {
      const response = await axios.post("http://localhost:8094/resolveTicket", {
        ticket_text: complaintText,
      });

      // Save the ticket to the context
      setTickets([...tickets, response.data]);

      setResponseInfo(response.data);
      setModalOpen(true);
    } catch (error) {
      console.error("Error submitting ticket:", error);
    }
  };

  return (
    <Container className="container" maxWidth="sm">
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          marginTop: 4,
        }}
      >
        <h1>User</h1>
        <TextField
          label="Complaint Text"
          multiline
          rows={4}
          fullWidth
          value={complaintText}
          onChange={(e) => setComplaintText(e.target.value)}
        />
        <Button variant="outlined" onClick={handleSubmit}>
          Submit Ticket
        </Button>
        <Button variant="outlined" disabled>
          Upload Attachments (Disabled)
        </Button>
        <Button variant="outlined" onClick={() => navigate('/')}>
          Back
        </Button>
        {modalOpen && (
          <Dialog
            open={modalOpen}
            onClose={() => setModalOpen(false)}
            maxWidth="sm"
            fullWidth
          >
            <DialogTitle>Response Information</DialogTitle>
            <DialogContent>
              <DialogContentText>
                {responseInfo.ticket_answer_translated}
              </DialogContentText>
            </DialogContent>
          </Dialog>
        )}
      </Box>
    </Container>
  );
};

export default AppUser;