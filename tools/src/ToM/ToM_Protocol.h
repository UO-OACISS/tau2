
class ToM_Protocol {
 public:
  virtual void registerProtocol() = 0;
  
  virtual void activateProtocol() = 0;
  virtual void deactivateProtocol() = 0;

  // Protocol implementations will host Protocol-specific filters here.

  // Protocol implementations will host Protocol-specific Streams here.

}
