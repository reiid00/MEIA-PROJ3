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
      <Box className="form-box">
        <h1>Support System - User Perspective</h1>
        <TextField
          label="Complaint Text"
          multiline
          rows={4}
          fullWidth
          value={complaintText}
          onChange={(e) => setComplaintText(e.target.value)}
        />
        <Button className="primary-button" onClick={handleSubmit}>
          Submit Ticket
        </Button>
        <Button className="secondary-button" disabled>
          Upload Attachments (Future Work)
        </Button>
        <Button className="back-button" onClick={() => navigate('/')}>
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
              <Typography variant="h6" component="div" gutterBottom>
                {responseInfo.ticket_answer_translated}
              </Typography>
            </DialogContent>
          </Dialog>
        )}
      </Box>
    </Container>
  );
};

export default AppUser;