import React, { useState, useContext } from 'react';
import {
  Container,
  Box,
  List,
  ListItem,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Typography,
} from '@mui/material';
import { TicketContext } from './TicketContext';
import { useNavigate } from 'react-router-dom';
import './App.css';

const AdminPage = () => {
  const [tickets] = useContext(TicketContext);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const navigate = useNavigate();

  const handleListItemClick = (ticket) => {
    setSelectedTicket(ticket);
  };

  const handleClose = () => {
    setSelectedTicket(null);
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
        <h1>Admin</h1>
        <List className="ticket-list">
          {tickets.map((ticket, index) => (
            <ListItem button key={index} onClick={() => handleListItemClick(ticket)}>
              <ListItemText primary={ticket.ticket_text} />
            </ListItem>
          ))}
        </List>
        {selectedTicket && (
          <Dialog
            open={Boolean(selectedTicket)}
            onClose={handleClose}
            maxWidth="sm"
            fullWidth
          >
            <DialogTitle>Ticket Details</DialogTitle>
            <DialogContent>
              <DialogContentText>
                <strong>Ticket Text:</strong> {selectedTicket.ticket_text}
              </DialogContentText>
              <DialogContentText>
                <strong>Ticket Text Translated:</strong> {selectedTicket.ticket_text_translated}
              </DialogContentText>
              <DialogContentText>
                <strong>Ticket Answer:</strong> {selectedTicket.ticket_answer}
              </DialogContentText>
              <DialogContentText>
                <strong>Ticket Answer Translated:</strong> {selectedTicket.ticket_answer_translated}
              </DialogContentText>
              <Typography variant="body1">
                <strong>Detected Language:</strong> {selectedTicket.detected_language}
              </Typography>
              <Typography variant="body1">
                <strong>Emotions:</strong> {selectedTicket.emotions}
              </Typography>
              <Typography variant="body1">
                <strong>Product:</strong> {selectedTicket.product}
              </Typography>
              <Typography variant="body1">
                <strong>Sub-Product:</strong> {selectedTicket.sub_product}
              </Typography>
              <Typography variant="body1">
                <strong>Issue:</strong> {selectedTicket.issue}
              </Typography>
              <Typography variant="body1">
                <strong>Sub-Issue:</strong> {selectedTicket.sub_issue}
              </Typography>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleClose} color="primary">
                Close
              </Button>
            </DialogActions>
          </Dialog>

            )}
            <Button
                variant="outlined"
                onClick={() => navigate("/")}
                sx={{ marginTop: 2 }}
            >
                Back
            </Button>
        </Box>
        
        </Container>
  );
};

export default AdminPage;