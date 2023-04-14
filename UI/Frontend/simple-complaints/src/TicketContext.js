import React, { createContext, useState } from "react";

export const TicketContext = createContext();

export const TicketProvider = ({ children }) => {
  const [tickets, setTickets] = useState([]);

  return (
    <TicketContext.Provider value={[tickets, setTickets]}>
      {children}
    </TicketContext.Provider>
  );
};