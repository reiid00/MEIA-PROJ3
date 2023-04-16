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
  Collapse,
} from '@mui/material';
import { TicketContext } from './TicketContext';
import { useNavigate } from 'react-router-dom';
import './App.css';

const AdminPage = () => {
  const [tickets] = useContext(TicketContext);
  const [selectedTicket, setSelectedTicket] = useState(null);
  const [openMoreInfo, setOpenMoreInfo] = useState(false);
  const navigate = useNavigate();

  const handleClickOpen = (ticket) => {
    setSelectedTicket(ticket);
  };

  const handleClose = () => {
    setSelectedTicket(null);
  };

  const handleMoreInfoClick = () => {
    setOpenMoreInfo((prevOpen) => !prevOpen);
  };

  return (
    <Container className="container" maxWidth="sm">
      <Box className="ticket-list-box">
        <h1>Support System - Admin Perspective</h1>
        <List className="ticket-list">
          {tickets.map((ticket, index) => (
            <ListItem button key={index} onClick={() => handleClickOpen(ticket)}>
              <ListItemText
                primary={ticket.ticket_text.slice(0, 50) + '...'}
                secondary={`Emotions: ${ticket.emotions} | Product: ${ticket.product}`}
              />
            </ListItem>
          ))}
        </List>
        <Button className="back-button" onClick={() => navigate('/')}>
          Back
        </Button>
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
              <Button onClick={handleMoreInfoClick}>
                {openMoreInfo ? 'Show Less' : 'Show More'}
              </Button>
              <Collapse in={openMoreInfo}>
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
              </Collapse>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleClose} color="primary">
                Close
              </Button>
            </DialogActions>
          </Dialog>
        )}
      </Box>
    </Container>
  );
};

export default AdminPage;