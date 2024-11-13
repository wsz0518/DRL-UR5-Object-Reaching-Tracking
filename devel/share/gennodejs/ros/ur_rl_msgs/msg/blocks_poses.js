// Auto-generated. Do not edit!

// (in-package ur_rl_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class blocks_poses {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.x = null;
      this.y = null;
      this.z = null;
    }
    else {
      if (initObj.hasOwnProperty('x')) {
        this.x = initObj.x
      }
      else {
        this.x = [];
      }
      if (initObj.hasOwnProperty('y')) {
        this.y = initObj.y
      }
      else {
        this.y = [];
      }
      if (initObj.hasOwnProperty('z')) {
        this.z = initObj.z
      }
      else {
        this.z = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type blocks_poses
    // Serialize message field [x]
    bufferOffset = _arraySerializer.float64(obj.x, buffer, bufferOffset, null);
    // Serialize message field [y]
    bufferOffset = _arraySerializer.float64(obj.y, buffer, bufferOffset, null);
    // Serialize message field [z]
    bufferOffset = _arraySerializer.float64(obj.z, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type blocks_poses
    let len;
    let data = new blocks_poses(null);
    // Deserialize message field [x]
    data.x = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [y]
    data.y = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [z]
    data.z = _arrayDeserializer.float64(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 8 * object.x.length;
    length += 8 * object.y.length;
    length += 8 * object.z.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'ur_rl_msgs/blocks_poses';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '615bdf149c8f78fd88df8f8f89a7e200';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # message type to describe 3-D position of the cylinder blocks
    # variable length array, length decided by topic /current_cylinder_blocks
    # to be published as a topic
    
    float64[] x  # x coordinate in the world
    float64[] y  # y coordinate in the world
    float64[] z  # z coordinate in the world
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new blocks_poses(null);
    if (msg.x !== undefined) {
      resolved.x = msg.x;
    }
    else {
      resolved.x = []
    }

    if (msg.y !== undefined) {
      resolved.y = msg.y;
    }
    else {
      resolved.y = []
    }

    if (msg.z !== undefined) {
      resolved.z = msg.z;
    }
    else {
      resolved.z = []
    }

    return resolved;
    }
};

module.exports = blocks_poses;
